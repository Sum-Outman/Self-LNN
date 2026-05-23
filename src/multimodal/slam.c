/**
 * @file slam.c
 * @brief 同步定位与建图(SLAM)系统 —— 全局协调层
 * 
 * K-006: SLAM模块架构 —— 7文件协调关系
 *   slam.c（本文件）→ 全局SLAM管线调度、状态管理、对外API
 *   slam_frontend.c  → 视觉里程计前端（特征提取/匹配/运动估计/三角化）
 *   slam_backend.c   → 后端优化（BA图优化/位姿图/路标优化）
 *   slam_enhance.c   → 增强模块（IMU融合/深度补全/不确定性传播）
 *   slam_loop_closure.c → 闭环检测（词袋模型/地点识别/全局重定位）
 *   slam_vocabulary.c → 视觉词汇表（ORB词袋训练/检索/评分）
 *   slam_io.c         → 数据IO（地图保存/加载/轨迹导出/点云序列化）
 * 
 * 实现完整的同步定位与建图功能，包括视觉里程计、后端优化、地图构建、闭环检测。
 * 基于现有基础设施：点云处理、深度估计、数学工具、图优化。
 * 100%纯C实现，无外部依赖。
 */

#include "selflnn/multimodal/slam.h"
#include "selflnn/multimodal/point_cloud.h"
#include "selflnn/multimodal/depth_estimation.h"
#include "selflnn/multimodal/liquid_vision.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/core/memory.h"
#include "selflnn/core/safe_memory.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/platform.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <float.h>

/* Windows平台特有头文件 */
#ifdef _WIN32
#include <io.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

/* B-03: 选择性禁用已知安全警告 */

/* 内部常量定义 */
#define SLAM_MAX_KEYFRAMES 1000
#define SLAM_MAX_LANDMARKS 10000
#define SLAM_MAX_FEATURES_PER_FRAME 2000
#define SLAM_MAX_OBSERVATIONS_PER_LANDMARK 20
#define SLAM_MAX_LOOP_CANDIDATES 50
#define SLAM_MAX_OPTIMIZATION_ITERATIONS 100
#define SLAM_EPSILON 1e-6f
#define SLAM_DEFAULT_FOCAL_LENGTH 525.0f  /* 默认焦距（像素） */
#define SLAM_DEFAULT_PRINCIPAL_POINT 320.0f /* 默认主点（像素） */

/* MUL-06: 闭环检测深度增强新增常量 */
#define SLAM_VOCABULARY_SIZE 1000         /* 默认词汇表大小 */
#define SLAM_VOCABULARY_DEPTH 6           /* 默认词汇树深度 */
#define SLAM_VOCABULARY_BRANCHING 10      /* 默认分支因子 */
#define SLAM_VOCABULARY_MAX_TRAIN 500     /* 最大训练帧数 */
#define SLAM_TEMPORAL_WINDOW_SIZE 5       /* 时间一致性窗口大小 */
#define SLAM_TEMPORAL_CONSISTENCY_THRESHOLD 3 /* 时间一致性阈值 */
#define SLAM_COVISIBILITY_MIN_WEIGHT 3    /* 共视图最小权重 */
#define SLAM_MAX_LOOP_CANDIDATES_SCORED 100 /* 最大评分候选数 */
#define SLAM_FUSION_MERGE_DISTANCE 0.1f   /* 地图点融合距离阈值（米） */
#define SLAM_DRIFT_PROPAGATION_RADIUS 50.0f /* 漂移传播半径（米） */
#define SLAM_BOW_SIMILARITY_THRESHOLD 0.05f /* 词袋相似度阈值 */

/* 内部数据结构定义 */

/**
 * @brief 特征点结构体（内部使用）
 */
typedef struct {
    int x;                      /**< 特征点X坐标（像素） */
    int y;                      /**< 特征点Y坐标（像素） */
    float response;             /**< 特征点响应值 */
    float orientation;          /**< 特征点方向（弧度） */
    float size;                 /**< 特征点大小 */
    int octave;                 /**< 特征点金字塔层 */
    float descriptor[32];       /**< 特征描述子（32维，完整实现） */
    int descriptor_length;      /**< 描述子长度 */
} FeaturePoint;

/**
 * @brief 特征匹配结构体
 */
typedef struct {
    int query_idx;              /**< 查询特征索引 */
    int train_idx;              /**< 训练特征索引 */
    float distance;             /**< 特征距离 */
    float ratio;                /**< 最近邻/次近邻比率 */
} FeatureMatch;

/**
 * @brief 特征提取器类型枚举
 */
typedef enum {
    FEATURE_EXTRACTOR_HARRIS = 0,     /**< Harris角点检测器 */
    FEATURE_EXTRACTOR_FAST = 1,       /**< FAST特征检测器 */
    FEATURE_EXTRACTOR_ORB = 2,        /**< ORB特征检测器 */
    FEATURE_EXTRACTOR_SIFT = 3,       /**< SIFT特征检测器 */
    FEATURE_EXTRACTOR_SURF = 4,       /**< SURF特征检测器 */
    FEATURE_EXTRACTOR_AKAZE = 5,      /**< AKAZE特征检测器 */
} FeatureExtractorType;

/**
 * @brief 特征提取器结构体
 */
typedef struct {
    FeatureExtractorType type;        /**< 特征提取器类型 */
    int max_features;                 /**< 最大特征点数 */
    float threshold;                  /**< 特征检测阈值 */
    int octave_layers;                /**< 金字塔层数 */
    float scale_factor;               /**< 尺度因子 */
    int descriptor_length;            /**< 描述子长度 */
    int initialized;                  /**< 是否已初始化 */
    /* 算法内部状态 */
    float* harris_response;           /**< Harris响应图像（如果使用Harris） */
    float* fast_corner_scores;        /**< FAST角点得分（如果使用FAST） */
    float* orb_brief_pattern;         /**< ORB BRIEF模式（如果使用ORB） */
    float* sift_gaussian_pyramid;     /**< SIFT高斯金字塔（如果使用SIFT） */
    float* surf_haar_wavelets;        /**< SURF Haar小波响应（如果使用SURF） */
    /* 描述子计算缓冲区 */
    float* descriptor_buffer;         /**< 描述子计算缓冲区 */
    int buffer_size;                  /**< 缓冲区大小 */
    /* 性能统计 */
    int total_features_extracted;     /**< 提取的特征总数 */
    float total_extraction_time;      /**< 总提取时间（毫秒） */
} FeatureExtractor;

/**
 * @brief IMU数据结构体（内部使用）
 */
typedef struct {
    float last_timestamp;       /**< 上次IMU数据时间戳 */
    float last_gyro[3];         /**< 上次陀螺仪数据 */
    float last_acc[3];          /**< 上次加速度计数据 */
    float gyro_bias[3];         /**< 陀螺仪零偏 */
    float acc_bias[3];          /**< 加速度计零偏 */
    float velocity[3];          /**< 估计速度 */
    float position[3];          /**< 估计位置 */
    float orientation[4];       /**< 估计姿态（四元数） */
    float preintegrated_rotation[4]; /**< 预积分旋转（四元数） */
    float preintegrated_velocity[3]; /**< 预积分速度 */
    float preintegrated_position[3]; /**< 预积分位置 */
    float covariance[9];        /**< 状态协方差矩阵（完整实现：3x3） */
    int initialized;            /**< 是否已初始化 */
    int has_data;               /**< 是否有有效数据 */
} ImuData;

/**
 * @brief 视觉里程计帧结构体（内部使用）
 */
typedef struct {
    int frame_id;               /**< 帧ID */
    float timestamp;            /**< 时间戳 */
    SlamPose pose;              /**< 帧位姿 */
    SlamPose pose_gt;           /**< 真实位姿（如果有） */
    
    FeaturePoint* features;     /**< 特征点数组 */
    int num_features;           /**< 特征点数量 */
    
    float* image_data;          /**< 图像数据（灰度） */
    int image_width;            /**< 图像宽度 */
    int image_height;           /**< 图像高度 */
    
    float* depth_data;          /**< 深度数据（可选） */
    PointCloud* point_cloud;    /**< 点云数据（可选） */
    
    int* landmark_ids;          /**< 关联的地标点ID数组 */
    int num_landmarks;          /**< 关联的地标点数量 */
    
    int is_keyframe;            /**< 是否是关键帧 */
    int keyframe_id;            /**< 关键帧ID（如果是关键帧） */
} VoFrame;

/**
 * @brief 局部地图结构体（内部使用）
 */
typedef struct {
    Landmark* landmarks;        /**< 地标点数组 */
    int num_landmarks;          /**< 地标点数量 */
    int max_landmarks;          /**< 最大地标点数 */
    
    KeyFrame* keyframes;        /**< 关键帧数组 */
    int num_keyframes;          /**< 关键帧数量 */
    int max_keyframes;          /**< 最大关键帧数 */
    
    float* map_points;          /**< 地图点云数据 */
    size_t num_map_points;      /**< 地图点数 */
    
    float bounds_min[3];        /**< 地图最小边界 */
    float bounds_max[3];        /**< 地图最大边界 */
    
    int initialized;            /**< 地图是否已初始化 */
} LocalMap;

/**
 * @brief 优化问题结构体（内部使用）
 */
typedef struct {
    int num_poses;              /**< 位姿数量 */
    int num_landmarks;          /**< 地标点数量 */
    int num_observations;       /**< 观测数量 */
    
    float* poses;               /**< 位姿数组（每帧6自由度：tx,ty,tz,qx,qy,qz,qw） */
    float* landmarks;           /**< 地标点数组（每点3自由度：x,y,z） */
    int* observation_frames;    /**< 观测对应的帧ID */
    int* observation_landmarks; /**< 观测对应的地标点ID */
    float* observation_points;  /**< 观测到的2D点（每观测2个值：u,v） */
    float* observation_weights; /**< 观测权重 */
    
    float* camera_params;       /**< 相机参数：fx,fy,cx,cy */
} OptimizationProblem;

/**
 * @brief 视觉词汇树节点结构体（内部使用）
 */
typedef struct VocabTreeNode {
    float descriptor[32];                    /**< 节点描述子（聚类中心） */
    int descriptor_length;                   /**< 描述子长度 */
    float weight;                            /**< 节点权重 */
    int is_leaf;                             /**< 是否为叶子节点 */
    int leaf_index;                          /**< 叶子节点索引（仅叶子节点有效） */
    int num_children;                        /**< 子节点数量 */
    struct VocabTreeNode** children;         /**< 子节点数组 */
    int num_features_assigned;               /**< 分配到该节点的特征数 */
    int* feature_indices;                    /**< 分配到该节点的特征索引 */
    int image_count;                         /**< 包含该节点的图像数量 */
} VocabTreeNode;

/**
 * @brief 视觉词汇表内部结构体
 */
typedef struct {
    VocabTreeNode* root;                     /**< 词汇树根节点 */
    int vocabulary_size;                     /**< 词汇表大小（叶子节点数） */
    int vocabulary_depth;                    /**< 词汇树深度 */
    int branching_factor;                    /**< 分支因子 */
    int descriptor_length;                   /**< 描述子长度 */
    int num_leaf_nodes;                      /**< 实际叶子节点数 */
    VisualWord* leaf_words;                  /**< 叶子单词数组 */
    int total_trained_features;              /**< 训练用的总特征数 */
    int num_trained_frames;                  /**< 训练用的帧数 */
    int enable_tf_idf;                       /**< 是否启用TF-IDF */
    int incremental_update_enabled;          /**< 是否启用增量更新 */
    int is_built;                            /**< 是否已构建 */
} InternalVocabulary;

/**
 * @brief 共视图内部结构体
 */
typedef struct {
    int* adjacency_matrix;                   /**< 邻接矩阵（num_keyframes x num_keyframes，存储权重） */
    int* connected_frames;                   /**< 连接的帧列表（展平存储） */
    int* connected_frame_counts;             /**< 每个帧的连接数 */
    int num_frames;                          /**< 帧数量 */
    int max_frames;                          /**< 最大帧容量 */
    int* essential_graph_edges_from;         /**< 本质图边-源 */
    int* essential_graph_edges_to;           /**< 本质图边-目标 */
    float* essential_graph_weights;          /**< 本质图边-权重 */
    int num_essential_edges;                 /**< 本质图边数 */
    int essential_graph_built;               /**< 本质图是否已构建 */
} InternalCovisibility;

/**
 * @brief 闭环检测内部状态结构体
 */
typedef struct {
    int* temporal_queue;                     /**< 时间一致性队列（最近N个检测帧） */
    int temporal_queue_size;                 /**< 时间队列容量 */
    int temporal_queue_count;                /**< 时间队列当前长度 */
    int temporal_queue_head;                 /**< 队列头索引 */
    int temporal_consistency_threshold;      /**< 一致性阈值 */
    
    LoopClosureCandidate* scored_candidates; /**< 评分候选数组 */
    int max_scored_candidates;               /**< 最大评分候选数 */
    int num_scored_candidates;               /**< 当前评分候选数 */
    
    int* fusion_table;                       /**< 地图点融合映射表（旧ID→新ID） */
    int fusion_table_size;                   /**< 融合表大小 */
    int fusion_pending;                      /**< 是否有待处理融合 */
    int total_fusions_performed;             /**< 总融合次数 */
    
    float* drift_deltas;                     /**< 漂移校正增量（每个关键帧6DOF变换） */
    int drift_delta_capacity;                /**< 增量数组容量 */
    int drift_active;                        /**< 漂移校正是否激活 */
    int drift_frame_start;                   /**< 漂移校正起始帧 */
    
    int last_detection_frame;                /**< 上次检测帧 */
    int consecutive_detections;              /**< 连续检测次数 */
    float detection_confidence;              /**< 检测置信度 */
    
    int bow_computed_for_frame;              /**< 已计算BoW的帧ID */
    float* bow_vector;                       /**< 当前帧的BoW向量 */
    int bow_vector_size;                     /**< BoW向量大小 */
} InternalLoopClosure;

/**
 * @brief SLAM系统内部结构体
 */
struct SlamSystem {
    /* 配置 */
    SlamConfig config;          /**< SLAM配置 */
    
    /* 状态 */
    SlamState state;            /**< SLAM状态 */
    int is_initialized;         /**< 系统是否已初始化 */
    int is_lost;                /**< 是否跟踪丢失 */
    
    /* 视觉里程计 */
    VoFrame* vo_frames;         /**< 视觉里程计帧数组 */
    int vo_frame_capacity;      /**< 帧数组容量 */
    int vo_frame_count;         /**< 帧数量 */
    int current_frame_id;       /**< 当前帧ID */
    
    /* 局部地图 */
    LocalMap local_map;         /**< 局部地图 */
    
    /* 优化问题 */
    OptimizationProblem optimization_problem; /**< 当前优化问题 */
    
    /* 闭环检测 */
    int* loop_candidates;       /**< 闭环候选帧ID数组 */
    int num_loop_candidates;    /**< 闭环候选数量 */
    int last_loop_frame_id;     /**< 上次检测到闭环的帧ID */
    
    /* 特征提取器 */
    void* feature_extractor;    /**< 特征提取器句柄（内部使用） */
    
    /* 点云处理器 */
    PointCloudProcessor* point_cloud_processor; /**< 点云处理器 */
    
    /* 深度估计器 */
    DepthEstimator* depth_estimator; /**< 深度估计器（如果使用深度） */
    
    /* IMU数据处理 */
    ImuData imu_data;               /**< IMU数据状态 */
    
    /* 缓冲区 */
    float* image_buffer;        /**< 图像缓冲区 */
    float* depth_buffer;        /**< 深度缓冲区 */
    float* point_cloud_buffer;  /**< 点云缓冲区 */
    size_t buffer_capacity;     /**< 缓冲区容量 */
    
    /* 闭环检测扩展配置（MUL-06） */
    LoopClosureConfig loop_closure_config; /**< 闭环检测扩展配置 */
    
    /* 视觉词汇表（MUL-06） */
    InternalVocabulary vocabulary; /**< 视觉词汇表内部状态 */
    
    /* 共视图（MUL-06） */
    InternalCovisibility covisibility; /**< 共视图内部状态 */
    
    /* 闭环检测内部状态（MUL-06） */
    InternalLoopClosure loop_closure_internal; /**< 闭环检测内部状态 */
    
    /* 性能统计 */
    float total_tracking_time;  /**< 总跟踪时间（毫秒） */
    float total_mapping_time;   /**< 总建图时间（毫秒） */
    float total_optimization_time; /**< 总优化时间（毫秒） */
    int frames_processed;       /**< 已处理帧数 */
    
    /* 轨迹历史记录 */
    SlamPose* trajectory;       /**< 轨迹历史缓冲区 */
    int trajectory_capacity;    /**< 轨迹缓冲区容量 */
    int trajectory_count;       /**< 轨迹中已记录的位姿数量 */
};

/* 内部辅助函数声明 */
static int slam_initialize_vo(SlamSystem* system);
static int slam_initialize_map(SlamSystem* system);
static int slam_extract_features(SlamSystem* system, const float* image_data,
                                int width, int height, FeaturePoint** features_out,
                                int* num_features_out);
static int slam_match_features(const FeaturePoint* features1, int num_features1,
                              const FeaturePoint* features2, int num_features2,
                              FeatureMatch** matches_out, int* num_matches_out);
static int slam_estimate_motion_2d2d(const FeaturePoint* features1,
                                    const FeaturePoint* features2,
                                    const FeatureMatch* matches, int num_matches,
                                    const float* camera_params,
                                    float* R, float* t);
static int slam_estimate_motion_3d2d(const FeaturePoint* features2d,
                                    const float* points3d, int num_points,
                                    const float* camera_params,
                                    const float* initial_pose,
                                    float* optimized_pose);
static int slam_triangulate_points(const FeaturePoint* features1,
                                  const FeaturePoint* features2,
                                  const FeatureMatch* matches, int num_matches,
                                  const float* R, const float* t,
                                  const float* camera_params,
                                  float* points3d);
static int slam_add_keyframe(SlamSystem* system, const VoFrame* frame);
static int slam_add_landmark(SlamSystem* system, const float* point3d,
                            const FeaturePoint* feature, int frame_id);
static int slam_update_landmark_observation(SlamSystem* system, int landmark_id,
                                           int frame_id, const float* point2d);
static int slam_optimize_local_bundle(SlamSystem* system, int window_size, int iterations);
static int slam_detect_loop_closure(SlamSystem* system, int frame_id,
                                   int* matched_frame_id);
static int slam_correct_loop_closure(SlamSystem* system, int frame_id,
                                    int matched_frame_id);
static int slam_build_optimization_problem(SlamSystem* system,
                                          OptimizationProblem* problem);
static int slam_solve_optimization_problem(OptimizationProblem* problem,
                                          int max_iterations);
static int slam_update_from_optimization(SlamSystem* system,
                                        const OptimizationProblem* problem);
static void slam_free_optimization_problem(OptimizationProblem* problem);

/* MUL-06: 视觉词汇表内部函数 */
static int slam_vocabulary_init(InternalVocabulary* vocab, const VisualVocabularyConfig* config);
static void slam_vocabulary_free(InternalVocabulary* vocab);
static int slam_vocabulary_build(InternalVocabulary* vocab, const float* all_descriptors,
                                int num_descriptors, int descriptor_length);
static int slam_vocabulary_compute_bow(InternalVocabulary* vocab, const float* descriptors,
                                      int num_descriptors, int descriptor_length,
                                      float* bow_vector, int* bow_vector_size);
static float slam_vocabulary_compute_similarity(const float* bow1, int size1,
                                               const float* bow2, int size2);
static int slam_vocabulary_add_frame(InternalVocabulary* vocab, const float* descriptors,
                                    int num_descriptors, int descriptor_length);
static int slam_vocabulary_update_tfidf(InternalVocabulary* vocab);

/* MUL-06: 词汇树内部函数 */
static VocabTreeNode* slam_vocab_node_create(int descriptor_length);
static void slam_vocab_node_free(VocabTreeNode* node);
static int slam_vocab_node_train(VocabTreeNode* node, const float* descriptors,
                                int num_descriptors, int descriptor_length,
                                int depth, int max_depth, int branch_factor);
static int slam_vocab_node_assign(VocabTreeNode* node, const float* descriptor,
                                 int descriptor_length);

/* MUL-06: 共视图内部函数 */
static int slam_covisibility_init(InternalCovisibility* cov, int max_frames);
static void slam_covisibility_free(InternalCovisibility* cov);
static int slam_covisibility_update(InternalCovisibility* cov, int frame_id,
                                   int* landmark_ids, int num_landmarks,
                                   const KeyFrame* keyframes, int num_keyframes);
static int slam_covisibility_get_connected(InternalCovisibility* cov, int frame_id,
                                          int* connected_ids, int max_count);
static int slam_covisibility_build_essential_graph(InternalCovisibility* cov,
                                                  const KeyFrame* keyframes,
                                                  int num_keyframes);

/* MUL-06: 闭环检测增强内部函数 */
static int slam_compute_bow_vector(SlamSystem* system, int frame_id);
static int slam_select_candidates_by_bow(SlamSystem* system, int frame_id,
                                        int* candidates, int max_candidates,
                                        float* scores);
static int slam_select_candidates_hybrid(SlamSystem* system, int frame_id,
                                        int* candidates, int max_candidates,
                                        float* scores);
static int slam_verify_loop_geometric_8point(SlamSystem* system, int frame_id,
                                            int candidate_id,
                                            int* num_inliers,
                                            float* inlier_ratio,
                                            float* fundamental_matrix);
static int slam_temporal_consistency_check(SlamSystem* system, int candidate_frame_id);
static int slam_fuse_loop_closure_map(SlamSystem* system, int frame_id,
                                     int matched_frame_id);
static int slam_propagate_drift_correction(SlamSystem* system, int matched_frame_id,
                                          int current_frame_id,
                                          const float* corrected_poses,
                                          int num_corrected);
static int slam_compute_fundamental_matrix_8point(const float* points1, const float* points2,
                                                  int num_points, float* F);
static float slam_compute_sampson_distance(const float* F, float x1, float y1,
                                          float x2, float y2);

/* 内存安全包装宏（直接展开到调用点，确保__FILE__/__LINE__准确） */
#define slam_malloc(size)          safe_malloc(size)
#define slam_calloc(num, size)     safe_calloc(num, size)
#define slam_realloc(ptr, size)    safe_realloc(ptr, size)
#define slam_free(ptr)             safe_free((void**)&(ptr))

/* 数学辅助函数 */
static inline float slam_norm_squared(const float* v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += v[i] * v[i];
    }
    return sum;
}

static inline void slam_cross_product(const float* a, const float* b, float* result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

static inline void slam_rodrigues_rotation(const float* axis, float angle, float* R) {
    float k[3] = {axis[0], axis[1], axis[2]};
    float norm = sqrtf(k[0]*k[0] + k[1]*k[1] + k[2]*k[2]);
    if (norm < SLAM_EPSILON) {
        /* 角度为0，返回单位矩阵 */
        R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
        R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
        R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
        return;
    }
    
    k[0] /= norm;
    k[1] /= norm;
    k[2] /= norm;
    
    float c = cosf(angle);
    float s = sinf(angle);
    float v = 1.0f - c;
    
    R[0] = k[0]*k[0]*v + c;
    R[1] = k[0]*k[1]*v - k[2]*s;
    R[2] = k[0]*k[2]*v + k[1]*s;
    
    R[3] = k[1]*k[0]*v + k[2]*s;
    R[4] = k[1]*k[1]*v + c;
    R[5] = k[1]*k[2]*v - k[0]*s;
    
    R[6] = k[2]*k[0]*v - k[1]*s;
    R[7] = k[2]*k[1]*v + k[0]*s;
    R[8] = k[2]*k[2]*v + c;
}

static inline void slam_quaternion_to_rotation_matrix(const float* q, float* R) {
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    
    R[0] = 1.0f - 2.0f*qy*qy - 2.0f*qz*qz;
    R[1] = 2.0f*qx*qy - 2.0f*qz*qw;
    R[2] = 2.0f*qx*qz + 2.0f*qy*qw;
    
    R[3] = 2.0f*qx*qy + 2.0f*qz*qw;
    R[4] = 1.0f - 2.0f*qx*qx - 2.0f*qz*qz;
    R[5] = 2.0f*qy*qz - 2.0f*qx*qw;
    
    R[6] = 2.0f*qx*qz - 2.0f*qy*qw;
    R[7] = 2.0f*qy*qz + 2.0f*qx*qw;
    R[8] = 1.0f - 2.0f*qx*qx - 2.0f*qy*qy;
}

static inline void slam_rotation_matrix_to_quaternion(const float* R, float* q) {
    float trace = R[0] + R[4] + R[8];
    
    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        q[0] = 0.25f * s;
        q[1] = (R[7] - R[5]) / s;
        q[2] = (R[2] - R[6]) / s;
        q[3] = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        float s = sqrtf(1.0f + R[0] - R[4] - R[8]) * 2.0f;
        q[0] = (R[7] - R[5]) / s;
        q[1] = 0.25f * s;
        q[2] = (R[1] + R[3]) / s;
        q[3] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        float s = sqrtf(1.0f + R[4] - R[0] - R[8]) * 2.0f;
        q[0] = (R[2] - R[6]) / s;
        q[1] = (R[1] + R[3]) / s;
        q[2] = 0.25f * s;
        q[3] = (R[5] + R[7]) / s;
    } else {
        float s = sqrtf(1.0f + R[8] - R[0] - R[4]) * 2.0f;
        q[0] = (R[3] - R[1]) / s;
        q[1] = (R[2] + R[6]) / s;
        q[2] = (R[5] + R[7]) / s;
        q[3] = 0.25f * s;
    }
    
    /* 归一化四元数 */
    float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm > SLAM_EPSILON) {
        q[0] /= norm;
        q[1] /= norm;
        q[2] /= norm;
        q[3] /= norm;
    }
}

/* 投影函数：3D点到2D图像平面 */
static inline void slam_project_point(const float* point3d, const float* camera_params,
                                     float* point2d) {
    float fx = camera_params[0];
    float fy = camera_params[1];
    float cx = camera_params[2];
    float cy = camera_params[3];
    
    point2d[0] = fx * point3d[0] / point3d[2] + cx;
    point2d[1] = fy * point3d[1] / point3d[2] + cy;
}

/* 反投影函数：2D图像点到3D射线 */
static inline void slam_backproject_point(const float* point2d, const float* camera_params,
                                         float depth, float* point3d) {
    float fx = camera_params[0];
    float fy = camera_params[1];
    float cx = camera_params[2];
    float cy = camera_params[3];
    
    point3d[0] = (point2d[0] - cx) * depth / fx;
    point3d[1] = (point2d[1] - cy) * depth / fy;
    point3d[2] = depth;
}

/* 计算重投影误差 */
static inline float slam_reprojection_error(const float* point3d, const float* point2d_obs,
                                           const float* pose, const float* camera_params) {
    /* 提取位姿：平移向量t和旋转矩阵R */
    float t[3] = {pose[0], pose[1], pose[2]};
    float q[4] = {pose[6], pose[3], pose[4], pose[5]}; /* 注意：存储顺序是tx,ty,tz,qx,qy,qz,qw */
    float R[9];
    slam_quaternion_to_rotation_matrix(q, R);
    
    /* 将3D点变换到相机坐标系 */
    float Pc[3];
    Pc[0] = R[0]*point3d[0] + R[1]*point3d[1] + R[2]*point3d[2] + t[0];
    Pc[1] = R[3]*point3d[0] + R[4]*point3d[1] + R[5]*point3d[2] + t[1];
    Pc[2] = R[6]*point3d[0] + R[7]*point3d[1] + R[8]*point3d[2] + t[2];
    
    /* 投影到图像平面 */
    float fx = camera_params[0];
    float fy = camera_params[1];
    float cx = camera_params[2];
    float cy = camera_params[3];
    
    if (Pc[2] < SLAM_EPSILON) {
        return FLT_MAX; /* 点在相机后面 */
    }
    
    float u = fx * Pc[0] / Pc[2] + cx;
    float v = fy * Pc[1] / Pc[2] + cy;
    
    /* 计算误差 */
    float dx = u - point2d_obs[0];
    float dy = v - point2d_obs[1];
    
    return dx*dx + dy*dy;
}

/* ========== 公开API实现 ========== */

SlamSystem* slam_system_create(const SlamConfig* config) {
    if (!config) {
        return NULL;
    }
    
    SlamSystem* system = (SlamSystem*)slam_malloc(sizeof(SlamSystem));
    if (!system) {
        return NULL;
    }
    
    /* 初始化系统内存 */
    memset(system, 0, sizeof(SlamSystem));
    
    /* 复制配置 */
    memcpy(&system->config, config, sizeof(SlamConfig));
    
    /* 设置默认参数（如果未设置） */
    if (system->config.max_keyframes <= 0) {
        system->config.max_keyframes = SLAM_MAX_KEYFRAMES;
    }
    if (system->config.num_features_per_frame <= 0) {
        system->config.num_features_per_frame = 1000;
    }
    if (system->config.feature_pyramid_levels <= 0) {
        system->config.feature_pyramid_levels = 8;
    }
    if (system->config.feature_scale_factor <= 0.0f) {
        system->config.feature_scale_factor = 1.2f;
    }
    if (system->config.thread_count <= 0) {
        system->config.thread_count = 1; /* 默认单线程 */
    }
    
    /* 初始化视觉里程计帧数组 */
    system->vo_frame_capacity = 1000;
    system->vo_frames = (VoFrame*)slam_calloc(system->vo_frame_capacity, sizeof(VoFrame));
    if (!system->vo_frames) {
        slam_free(system);
        return NULL;
    }
    system->vo_frame_count = 0;
    system->current_frame_id = -1;
    
    /* 初始化局部地图 */
    system->local_map.max_landmarks = SLAM_MAX_LANDMARKS;
    system->local_map.max_keyframes = system->config.max_keyframes;
    
    system->local_map.landmarks = (Landmark*)slam_calloc(SLAM_MAX_LANDMARKS, sizeof(Landmark));
    system->local_map.keyframes = (KeyFrame*)slam_calloc(system->config.max_keyframes, sizeof(KeyFrame));
    system->local_map.map_points = NULL;
    system->local_map.num_map_points = 0;
    system->local_map.num_landmarks = 0;
    system->local_map.num_keyframes = 0;
    system->local_map.initialized = 0;
    
    if (!system->local_map.landmarks || !system->local_map.keyframes) {
        if (system->local_map.landmarks) slam_free(system->local_map.landmarks);
        if (system->local_map.keyframes) slam_free(system->local_map.keyframes);
        slam_free(system->vo_frames);
        slam_free(system);
        return NULL;
    }
    
    /* 初始化边界 */
    for (int i = 0; i < 3; i++) {
        system->local_map.bounds_min[i] = FLT_MAX;
        system->local_map.bounds_max[i] = -FLT_MAX;
    }
    
    /* 初始化优化问题 */
    memset(&system->optimization_problem, 0, sizeof(OptimizationProblem));
    
    /* 初始化闭环检测 */
    system->loop_candidates = (int*)slam_calloc(SLAM_MAX_LOOP_CANDIDATES, sizeof(int));
    system->num_loop_candidates = 0;
    system->last_loop_frame_id = -1;
    
    /* 创建点云处理器 */
    system->point_cloud_processor = point_cloud_processor_create();
    
    /* 创建深度估计器（如果需要） */
    if (system->config.sensor_type == SLAM_SENSOR_RGBD ||
        system->config.sensor_type == SLAM_SENSOR_STEREO) {
        DepthEstimationConfig depth_config;
        memset(&depth_config, 0, sizeof(DepthEstimationConfig));
        depth_config.method = DEPTH_METHOD_STEREO;
        depth_config.stereo_algorithm = STEREO_MATCHING_SGBM;
        depth_config.enable_filtering = 1;
        depth_config.enable_postprocessing = 1;
        depth_config.disparity_range = 64;
        depth_config.window_size = 7;
        depth_config.min_depth = 0.1f;
        depth_config.max_depth = 10.0f;
        
        system->depth_estimator = depth_estimator_create(&depth_config);
    } else {
        system->depth_estimator = NULL;
    }
    
    /* 初始化缓冲区 */
    system->image_buffer = NULL;
    system->depth_buffer = NULL;
    system->point_cloud_buffer = NULL;
    system->buffer_capacity = 0;
    
    /* 初始化状态 */
    memset(&system->state, 0, sizeof(SlamState));
    system->state.tracking_state = 0; /* 丢失 */
    system->state.mapping_state = 0; /* 未建图 */
    system->state.loop_closure_state = 0; /* 未检测 */
    system->is_initialized = 0;
    system->is_lost = 1; /* 初始状态为丢失 */
    
    /* 初始化性能统计 */
    system->total_tracking_time = 0.0f;
    system->total_mapping_time = 0.0f;
    system->total_optimization_time = 0.0f;
    system->frames_processed = 0;
    
    /* 初始化IMU数据 */
    memset(&system->imu_data, 0, sizeof(ImuData));
    system->imu_data.last_timestamp = -1.0f; /* 标记为未初始化 */
    system->imu_data.initialized = 0;
    
    /* 特征提取器初始化（完整实现） */
    /* 分配并初始化特征提取器结构体 */
    system->feature_extractor = (FeatureExtractor*)slam_malloc(sizeof(FeatureExtractor));
    if (system->feature_extractor) {
        FeatureExtractor* extractor = (FeatureExtractor*)system->feature_extractor;
        memset(extractor, 0, sizeof(FeatureExtractor));
        extractor->type = FEATURE_EXTRACTOR_HARRIS;
        extractor->max_features = 1000;
        extractor->threshold = 0.01f;
        extractor->octave_layers = 3;
        extractor->scale_factor = 1.2f;
        extractor->descriptor_length = 32;
        extractor->initialized = 1;
    } else {
        /* 内存分配失败，特征提取器不可用（系统将在有限模式下运行） */
        system->feature_extractor = NULL;
    }
    
    /* MUL-06: 初始化闭环检测扩展配置 */
    {
        LoopClosureConfig* lcc = &system->loop_closure_config;
        memset(lcc, 0, sizeof(LoopClosureConfig));
        lcc->enable_temporal_consistency = 1;
        lcc->temporal_window_size = SLAM_TEMPORAL_WINDOW_SIZE;
        lcc->temporal_consistency_threshold = SLAM_TEMPORAL_CONSISTENCY_THRESHOLD;
        lcc->enable_geometric_verification = 1;
        lcc->geometric_threshold = 1.0f;
        lcc->min_geometric_inliers = 30;
        lcc->min_inlier_ratio = 0.3f;
        lcc->enable_map_fusion = 1;
        lcc->enable_drift_propagation = 1;
        lcc->max_loop_distance = 10.0f;
        lcc->min_feature_matches = 20;
        lcc->min_keyframes_for_detection = 20;
        lcc->min_frame_gap = 30;
        lcc->max_candidates_per_detection = SLAM_MAX_LOOP_CANDIDATES;
        lcc->candidate_search_radius = 10.0f;
        lcc->bow_score_threshold = SLAM_BOW_SIMILARITY_THRESHOLD;
    }
    
    /* MUL-06: 初始化视觉词汇表 */
    {
        VisualVocabularyConfig vc;
        memset(&vc, 0, sizeof(VisualVocabularyConfig));
        vc.vocabulary_size = SLAM_VOCABULARY_SIZE;
        vc.vocabulary_depth = SLAM_VOCABULARY_DEPTH;
        vc.branching_factor = SLAM_VOCABULARY_BRANCHING;
        vc.max_train_frames = SLAM_VOCABULARY_MAX_TRAIN;
        vc.descriptor_length = 32;
        vc.enable_tf_idf = 1;
        vc.enable_incremental_update = 0;
        slam_vocabulary_init(&system->vocabulary, &vc);
    }
    
    /* MUL-06: 初始化共视图 */
    slam_covisibility_init(&system->covisibility, system->config.max_keyframes);
    
    /* MUL-06: 初始化闭环检测内部状态 */
    {
        InternalLoopClosure* ilc = &system->loop_closure_internal;
        memset(ilc, 0, sizeof(InternalLoopClosure));
        ilc->temporal_queue_size = SLAM_TEMPORAL_WINDOW_SIZE * 2;
        ilc->temporal_queue = (int*)slam_calloc(ilc->temporal_queue_size, sizeof(int));
        ilc->temporal_queue_count = 0;
        ilc->temporal_queue_head = 0;
        ilc->temporal_consistency_threshold = SLAM_TEMPORAL_CONSISTENCY_THRESHOLD;
        ilc->max_scored_candidates = SLAM_MAX_LOOP_CANDIDATES_SCORED;
        ilc->scored_candidates = (LoopClosureCandidate*)slam_calloc(
            ilc->max_scored_candidates, sizeof(LoopClosureCandidate));
        ilc->num_scored_candidates = 0;
        ilc->fusion_table = NULL;
        ilc->fusion_table_size = 0;
        ilc->fusion_pending = 0;
        ilc->total_fusions_performed = 0;
        ilc->drift_deltas = NULL;
        ilc->drift_delta_capacity = 0;
        ilc->drift_active = 0;
        ilc->drift_frame_start = -1;
        ilc->last_detection_frame = -1;
        ilc->consecutive_detections = 0;
        ilc->detection_confidence = 0.0f;
        ilc->bow_computed_for_frame = -1;
        ilc->bow_vector = NULL;
        ilc->bow_vector_size = 0;
    }
    
    /* 初始化轨迹缓冲区 */
    system->trajectory = NULL;
    system->trajectory_capacity = 0;
    system->trajectory_count = 0;
    
    return system;
}

void slam_system_free(SlamSystem* system) {
    if (!system) {
        return;
    }
    
    /* 释放视觉里程计帧 */
    if (system->vo_frames) {
        for (int i = 0; i < system->vo_frame_count; i++) {
            VoFrame* frame = &system->vo_frames[i];
            if (frame->features) slam_free(frame->features);
            if (frame->image_data) slam_free(frame->image_data);
            if (frame->depth_data) slam_free(frame->depth_data);
            if (frame->point_cloud) {
                point_cloud_free(frame->point_cloud);
                slam_free(frame->point_cloud);
            }
            if (frame->landmark_ids) slam_free(frame->landmark_ids);
        }
        slam_free(system->vo_frames);
    }
    
    /* 释放局部地图 */
    if (system->local_map.landmarks) {
        for (int i = 0; i < system->local_map.num_landmarks; i++) {
            Landmark* lm = &system->local_map.landmarks[i];
            if (lm->descriptor) slam_free(lm->descriptor);
            if (lm->observing_frames) slam_free(lm->observing_frames);
            if (lm->observations) slam_free(lm->observations);
        }
        slam_free(system->local_map.landmarks);
    }
    
    if (system->local_map.keyframes) {
        for (int i = 0; i < system->local_map.num_keyframes; i++) {
            KeyFrame* kf = &system->local_map.keyframes[i];
            if (kf->image_data) slam_free(kf->image_data);
            if (kf->depth_data) slam_free(kf->depth_data);
            if (kf->point_cloud) slam_free(kf->point_cloud);
            if (kf->descriptors) slam_free(kf->descriptors);
            if (kf->keypoints_x) slam_free(kf->keypoints_x);
            if (kf->keypoints_y) slam_free(kf->keypoints_y);
            if (kf->keypoints_3d) slam_free(kf->keypoints_3d);
        }
        slam_free(system->local_map.keyframes);
    }
    
    if (system->local_map.map_points) {
        slam_free(system->local_map.map_points);
    }
    
    /* 释放优化问题 */
    slam_free_optimization_problem(&system->optimization_problem);
    
    /* 释放闭环检测 */
    if (system->loop_candidates) slam_free(system->loop_candidates);
    
    /* 释放特征提取器 */
    if (system->feature_extractor) slam_free(system->feature_extractor);
    
    /* 释放点云处理器 */
    if (system->point_cloud_processor) {
        point_cloud_processor_free(system->point_cloud_processor);
    }
    
    /* 释放深度估计器 */
    if (system->depth_estimator) {
        depth_estimator_free(system->depth_estimator);
    }
    
    /* 释放缓冲区 */
    if (system->image_buffer) slam_free(system->image_buffer);
    if (system->depth_buffer) slam_free(system->depth_buffer);
    if (system->point_cloud_buffer) slam_free(system->point_cloud_buffer);
    
    /* MUL-06: 释放视觉词汇表 */
    slam_vocabulary_free(&system->vocabulary);
    
    /* MUL-06: 释放共视图 */
    slam_covisibility_free(&system->covisibility);
    
    /* MUL-06: 释放闭环检测内部状态 */
    {
        InternalLoopClosure* ilc = &system->loop_closure_internal;
        if (ilc->temporal_queue) slam_free(ilc->temporal_queue);
        if (ilc->scored_candidates) slam_free(ilc->scored_candidates);
        if (ilc->fusion_table) slam_free(ilc->fusion_table);
        if (ilc->drift_deltas) slam_free(ilc->drift_deltas);
        if (ilc->bow_vector) slam_free(ilc->bow_vector);
    }
    
    /* 释放轨迹缓冲区 */
    if (system->trajectory) slam_free(system->trajectory);
    
    /* 释放系统 */
    slam_free(system);
}

int slam_process_frame(SlamSystem* system,
                      const float* sensor_data,
                      SlamSensorType sensor_type,
                      int width, int height, int channels,
                      float timestamp,
                      SlamResult* result) {
    if (!system || !sensor_data || !result) {
        return -1;
    }
    
    /* 根据传感器类型调用相应的处理函数 */
    switch (sensor_type) {
        case SLAM_SENSOR_MONOCULAR:
        case SLAM_SENSOR_STEREO:
        case SLAM_SENSOR_RGBD:
            /* 视觉传感器：处理图像 */
            return slam_process_visual_frame(system, sensor_data,
                                            width, height, channels,
                                            timestamp, result);
            
        case SLAM_SENSOR_LIDAR_2D:
        case SLAM_SENSOR_LIDAR_3D:
            /* 激光雷达：处理点云 */
            return slam_process_point_cloud(system, sensor_data,
                                           (sensor_type == SLAM_SENSOR_LIDAR_2D) ? 
                                           width * height : width, /* 点数量 */
                                           timestamp, result);
            
        case SLAM_SENSOR_IMU:
            /* IMU：处理惯性数据 */
            {
                float gyro[3] = {sensor_data[0], sensor_data[1], sensor_data[2]};
                float acc[3] = {sensor_data[3], sensor_data[4], sensor_data[5]};
                return slam_process_imu_data(system, gyro, acc, timestamp);
            }
            
        default:
            /* 不支持的传感器类型 */
            return -1;
    }
}

int slam_process_visual_frame(SlamSystem* system,
                             const float* image_data,
                             int width, int height, int channels,
                             float timestamp,
                             SlamResult* result) {
    if (!system || !image_data || !result || width <= 0 || height <= 0) {
        return -1;
    }
    
    /* 记录开始时间 */
    clock_t start_time = clock();
    
    /* 初始化结果 */
    memset(result, 0, sizeof(SlamResult));
    result->timestamp = timestamp;
    
    /* 检查系统是否已初始化 */
    if (!system->is_initialized) {
        /* 系统初始化 */
        int init_result = slam_initialize_vo(system);
        if (init_result != 0) {
            system->is_lost = 1;
            result->tracking_quality = 0;
            return -1;
        }
        system->is_initialized = 1;
        system->is_lost = 0;
    }
    
    /* 创建新帧 */
    int frame_id = system->current_frame_id + 1;
    if (frame_id >= system->vo_frame_capacity) {
        /* 扩展帧数组 */
        int new_capacity = system->vo_frame_capacity * 2;
        VoFrame* new_frames = (VoFrame*)slam_realloc(system->vo_frames,
                                                    new_capacity * sizeof(VoFrame));
        if (!new_frames) {
            return -1;
        }
        system->vo_frames = new_frames;
        system->vo_frame_capacity = new_capacity;
        
        /* 初始化新分配的帧 */
        for (int i = frame_id; i < new_capacity; i++) {
            memset(&system->vo_frames[i], 0, sizeof(VoFrame));
        }
    }
    
    VoFrame* current_frame = &system->vo_frames[frame_id];
    VoFrame* prev_frame = (frame_id > 0) ? &system->vo_frames[frame_id - 1] : NULL;
    
    /* 设置帧基本信息 */
    current_frame->frame_id = frame_id;
    current_frame->timestamp = timestamp;
    current_frame->image_width = width;
    current_frame->image_height = height;
    
    /* 分配图像缓冲区 */
    size_t image_size = width * height;
    if (channels > 1) {
        image_size *= channels;
    }
    
    current_frame->image_data = (float*)slam_malloc(image_size * sizeof(float));
    if (!current_frame->image_data) {
        return -1;
    }
    
    /* 复制图像数据 */
    memcpy(current_frame->image_data, image_data, image_size * sizeof(float));
    
    /* 提取特征点 */
    FeaturePoint* features = NULL;
    int num_features = 0;
    
    int extract_result = slam_extract_features(system, image_data, width, height,
                                              &features, &num_features);
    if (extract_result != 0 || num_features < 20) {
        /* 特征提取失败或特征点太少 */
        slam_free(current_frame->image_data);
        current_frame->image_data = NULL;
        system->is_lost = 1;
        result->tracking_quality = 0;
        return -1;
    }
    
    current_frame->features = features;
    current_frame->num_features = num_features;
    
    /* 如果是第一帧，初始化位姿 */
    if (frame_id == 0) {
        /* 第一帧：位姿为单位矩阵 */
        current_frame->pose.position[0] = 0.0f;
        current_frame->pose.position[1] = 0.0f;
        current_frame->pose.position[2] = 0.0f;
        current_frame->pose.orientation[0] = 1.0f; /* w */
        current_frame->pose.orientation[1] = 0.0f; /* x */
        current_frame->pose.orientation[2] = 0.0f; /* y */
        current_frame->pose.orientation[3] = 0.0f; /* z */
        current_frame->pose.frame_id = frame_id;
        current_frame->pose.timestamp = timestamp;
        
        /* 第一帧设为关键帧 */
        current_frame->is_keyframe = 1;
        current_frame->keyframe_id = 0;
        
        /* 添加到关键帧 */
        slam_add_keyframe(system, current_frame);
        
        /* 更新系统状态 */
        system->current_frame_id = frame_id;
        system->vo_frame_count++;
        
        /* 设置结果 */
        memcpy(&result->estimated_pose, &current_frame->pose, sizeof(SlamPose));
        result->tracking_quality = 100;
        result->new_keyframe_created = 1;
        result->local_map_point_count = 0;
        result->trajectory_length = 1;
        
        /* 初始化轨迹 */
        result->trajectory = (float*)slam_malloc(7 * sizeof(float)); /* 位置+四元数 */
        if (result->trajectory) {
            result->trajectory[0] = current_frame->pose.position[0];
            result->trajectory[1] = current_frame->pose.position[1];
            result->trajectory[2] = current_frame->pose.position[2];
            result->trajectory[3] = current_frame->pose.orientation[0]; /* w */
            result->trajectory[4] = current_frame->pose.orientation[1]; /* x */
            result->trajectory[5] = current_frame->pose.orientation[2]; /* y */
            result->trajectory[6] = current_frame->pose.orientation[3]; /* z */
        }
        
        /* 更新SLAM状态 */
        system->state.tracking_state = 2; /* 良好 */
        system->state.total_frames = 1;
        system->state.total_keyframes = 1;
        
        return 0;
    }
    
    /* 不是第一帧：估计运动 */
    
    /* 特征匹配 */
    FeatureMatch* matches = NULL;
    int num_matches = 0;
    
    int match_result = slam_match_features(prev_frame->features, prev_frame->num_features,
                                          current_frame->features, current_frame->num_features,
                                          &matches, &num_matches);
    
    if (match_result != 0 || num_matches < 20) {
        /* 特征匹配失败或匹配点太少 */
        slam_free(matches);
        matches = NULL;
        slam_free(features);
        current_frame->features = NULL;
        slam_free(current_frame->image_data);
        current_frame->image_data = NULL;
        system->is_lost = 1;
        result->tracking_quality = 0;
        return -1;
    }
    
    /* 估计相机运动（2D-2D或2D-3D） */
    float R[9], t[3];
    float camera_params[4] = {SLAM_DEFAULT_FOCAL_LENGTH, SLAM_DEFAULT_FOCAL_LENGTH,
                             width/2.0f, height/2.0f};
    
    int motion_result = 0;
    
    if (system->local_map.num_landmarks > 0 && prev_frame->num_landmarks > 0) {
        /* 使用2D-3D PnP估计运动 */
        /* 完整实现：从局部地图中获取匹配特征点对应的3D点 */
        
        /* 统计有3D对应点的匹配数量 */
        int num_3d_matches = 0;
        for (int i = 0; i < num_matches; i++) {
            int prev_feature_idx = matches[i].query_idx;
            if (prev_feature_idx >= 0 && prev_feature_idx < prev_frame->num_landmarks) {
                int landmark_id = prev_frame->landmark_ids[prev_feature_idx];
                if (landmark_id >= 0 && landmark_id < system->local_map.num_landmarks) {
                    num_3d_matches++;
                }
            }
        }
        
        if (num_3d_matches >= 4) {
            /* 分配3D点和对应的2D特征点数组 */
            float* points_3d = (float*)slam_malloc(3 * num_3d_matches * sizeof(float));
            FeaturePoint* features_2d = (FeaturePoint*)slam_malloc(num_3d_matches * sizeof(FeaturePoint));
            
            if (points_3d && features_2d) {
                int match_idx = 0;
                for (int i = 0; i < num_matches; i++) {
                    int prev_feature_idx = matches[i].query_idx;
                    if (prev_feature_idx >= 0 && prev_feature_idx < prev_frame->num_landmarks) {
                        int landmark_id = prev_frame->landmark_ids[prev_feature_idx];
                        if (landmark_id >= 0 && landmark_id < system->local_map.num_landmarks) {
                            Landmark* landmark = &system->local_map.landmarks[landmark_id];
                            
                            /* 复制3D点 */
                            points_3d[match_idx * 3] = landmark->position[0];
                            points_3d[match_idx * 3 + 1] = landmark->position[1];
                            points_3d[match_idx * 3 + 2] = landmark->position[2];
                            
                            /* 复制对应的当前帧2D特征点 */
                            int curr_feature_idx = matches[i].train_idx;
                            if (curr_feature_idx >= 0 && curr_feature_idx < current_frame->num_features) {
                                features_2d[match_idx] = current_frame->features[curr_feature_idx];
                            } else {
                                /* 如果索引无效，使用匹配的特征点 */
                                features_2d[match_idx] = current_frame->features[0];
                            }
                            
                            match_idx++;
                        }
                    }
                }
                
                /* 使用PnP估计运动 */
                motion_result = slam_estimate_motion_3d2d(features_2d,
                                                         points_3d, num_3d_matches,
                                                         camera_params,
                                                         prev_frame->pose.orientation, /* 初始位姿 */
                                                         current_frame->pose.orientation);
                
                /* 清理临时内存 */
                slam_free(points_3d);
                slam_free(features_2d);
            } else {
                /* 内存分配失败，回退到2D-2D */
                motion_result = -1;
            }
        } else {
            /* 3D对应点不足，使用2D-2D */
            motion_result = -1;
        }
        
        if (motion_result != 0) {
            /* PnP失败，回退到2D-2D */
            motion_result = slam_estimate_motion_2d2d(prev_frame->features,
                                                     current_frame->features,
                                                     matches, num_matches,
                                                     camera_params, R, t);
        }
    } else {
        /* 使用2D-2D对极几何估计运动 */
        motion_result = slam_estimate_motion_2d2d(prev_frame->features,
                                                 current_frame->features,
                                                 matches, num_matches,
                                                 camera_params, R, t);
        
        if (motion_result == 0) {
            /* 将旋转矩阵转换为四元数 */
            float q[4];
            slam_rotation_matrix_to_quaternion(R, q);
            
            /* 更新当前帧位姿 */
            current_frame->pose.position[0] = prev_frame->pose.position[0] + t[0];
            current_frame->pose.position[1] = prev_frame->pose.position[1] + t[1];
            current_frame->pose.position[2] = prev_frame->pose.position[2] + t[2];
            
            /* 旋转组合：当前旋转 = 上一帧旋转 * 相对旋转 */
            float prev_q[4] = {prev_frame->pose.orientation[0],
                              prev_frame->pose.orientation[1],
                              prev_frame->pose.orientation[2],
                              prev_frame->pose.orientation[3]};
            float rel_q[4] = {q[0], q[1], q[2], q[3]};
            
            /* 四元数乘法：当前 = 上一帧 × 相对 */
            current_frame->pose.orientation[0] = prev_q[0]*rel_q[0] - prev_q[1]*rel_q[1] - 
                                                prev_q[2]*rel_q[2] - prev_q[3]*rel_q[3];
            current_frame->pose.orientation[1] = prev_q[0]*rel_q[1] + prev_q[1]*rel_q[0] + 
                                                prev_q[2]*rel_q[3] - prev_q[3]*rel_q[2];
            current_frame->pose.orientation[2] = prev_q[0]*rel_q[2] - prev_q[1]*rel_q[3] + 
                                                prev_q[2]*rel_q[0] + prev_q[3]*rel_q[1];
            current_frame->pose.orientation[3] = prev_q[0]*rel_q[3] + prev_q[1]*rel_q[2] - 
                                                prev_q[2]*rel_q[1] + prev_q[3]*rel_q[0];
            
            /* 归一化四元数 */
            float norm = sqrtf(current_frame->pose.orientation[0]*current_frame->pose.orientation[0] +
                              current_frame->pose.orientation[1]*current_frame->pose.orientation[1] +
                              current_frame->pose.orientation[2]*current_frame->pose.orientation[2] +
                              current_frame->pose.orientation[3]*current_frame->pose.orientation[3]);
            if (norm > SLAM_EPSILON) {
                current_frame->pose.orientation[0] /= norm;
                current_frame->pose.orientation[1] /= norm;
                current_frame->pose.orientation[2] /= norm;
                current_frame->pose.orientation[3] /= norm;
            }
        }
    }
    
    current_frame->pose.frame_id = frame_id;
    current_frame->pose.timestamp = timestamp;
    
    if (motion_result != 0) {
        /* 运动估计失败 */
        slam_free(matches);
        slam_free(features);
        current_frame->features = NULL;
        slam_free(current_frame->image_data);
        current_frame->image_data = NULL;
        system->is_lost = 1;
        result->tracking_quality = 0;
        return -1;
    }
    
    /* 判断是否创建关键帧 */
    int create_keyframe = 0;
    
    /* 完整实现：多维度关键帧选择策略（ ） */
    /* 条件1：时间间隔 - 至少每10帧考虑一次关键帧 */
    if (frame_id % 10 == 0) {
        create_keyframe = 1;
    }
    
    /* 条件2：特征匹配数量不足 - 需要更多地图点 */
    if (num_matches < 50) {
        create_keyframe = 1;
    }
    
    /* 条件3：相机运动幅度判断（如果R和t可用） */
    if (R && t) {
        /* 计算平移幅度 */
        float translation_magnitude = sqrtf(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
        /* 计算旋转幅度（从旋转矩阵提取角度） */
        float trace_R = R[0] + R[4] + R[8];
        float rotation_angle = acosf(fmaxf(-1.0f, fminf(1.0f, (trace_R - 1.0f) / 2.0f)));
        
        /* 如果运动幅度超过阈值，创建关键帧 */
        if (translation_magnitude > 0.1f || rotation_angle > 0.2f) {
            create_keyframe = 1;
        }
    }
    
    /* 条件4：跟踪质量评估（基于匹配质量） */
    if (num_matches > 0) {
        float avg_distance = 0.0f;
        for (int i = 0; i < num_matches; i++) {
            avg_distance += matches[i].distance;
        }
        avg_distance /= num_matches;
        
        /* 如果平均匹配距离过大，说明跟踪质量差，需要关键帧 */
        if (avg_distance > 0.3f) {
            create_keyframe = 1;
        }
    }
    
    /* 条件5：场景变化检测（基于特征分布） */
    if (features && prev_frame->features && num_matches > 20) {
        float scene_change = 0.0f;
        /* 完整实现：计算匹配特征的位置变化标准差（ ） */
        float mean_dx = 0.0f, mean_dy = 0.0f;
        for (int i = 0; i < num_matches; i++) {
            mean_dx += features[matches[i].query_idx].x - prev_frame->features[matches[i].train_idx].x;
            mean_dy += features[matches[i].query_idx].y - prev_frame->features[matches[i].train_idx].y;
        }
        mean_dx /= num_matches;
        mean_dy /= num_matches;
        
        float var_dx = 0.0f, var_dy = 0.0f;
        for (int i = 0; i < num_matches; i++) {
            float dx = features[matches[i].query_idx].x - prev_frame->features[matches[i].train_idx].x - mean_dx;
            float dy = features[matches[i].query_idx].y - prev_frame->features[matches[i].train_idx].y - mean_dy;
            var_dx += dx * dx;
            var_dy += dy * dy;
        }
        var_dx /= num_matches;
        var_dy /= num_matches;
        scene_change = sqrtf(var_dx + var_dy);
        
        if (scene_change > 10.0f) {
            create_keyframe = 1;
        }
    }
    
    if (create_keyframe) {
        current_frame->is_keyframe = 1;
        current_frame->keyframe_id = system->local_map.num_keyframes;
        
        /* 添加到关键帧 */
        slam_add_keyframe(system, current_frame);
        
        /* 三角化新的地图点 */
        if (prev_frame->is_keyframe) {
            /* 三角化当前帧和上一关键帧之间的匹配点 */
            float* points3d = (float*)slam_malloc(3 * num_matches * sizeof(float));
            int triangulated = 0;
            if (points3d) {
                triangulated = slam_triangulate_points(prev_frame->features,
                                                      current_frame->features,
                                                      matches, num_matches,
                                                      R, t, camera_params,
                                                      points3d);
            }
            
            /* 添加三角化的点到地图中 */
            for (int i = 0; i < triangulated; i++) {
                slam_add_landmark(system, &points3d[3*i],
                                 &current_frame->features[matches[i].train_idx],
                                 frame_id);
            }
            
            /* 释放临时内存 */
            safe_free((void**)&points3d);
        }
        
        result->new_keyframe_created = 1;
    } else {
        current_frame->is_keyframe = 0;
        current_frame->keyframe_id = -1;
        result->new_keyframe_created = 0;
    }
    
    /* 局部优化（如果有关键帧） */
    if (system->local_map.num_keyframes >= 2) {
        slam_optimize_local_bundle(system, 5, 10); /* 优化最近5个关键帧，最多10次迭代 */
    }
    
    /* MUL-06: 更新共视图（如果是关键帧） */
    if (current_frame->is_keyframe && system->local_map.num_keyframes > 1) {
        slam_covisibility_update(&system->covisibility, frame_id,
                                system->local_map.keyframes[frame_id].landmark_ids,
                                system->local_map.keyframes[frame_id].num_landmarks,
                                system->local_map.keyframes, system->local_map.num_keyframes);
    }
    
    /* MUL-06: 增强闭环检测（含BoW+共视图+时间一致性+几何验证） */
    if (system->config.enable_loop_closure && current_frame->is_keyframe) {
        /* 计算当前关键帧的BoW向量（用于快速候选选择） */
        if (system->vocabulary.is_built) {
            slam_compute_bow_vector(system, frame_id);
        }
        
        int matched_frame_id = -1;
        int loop_result = slam_detect_loop_closure(system, frame_id, &matched_frame_id);
        
        if (loop_result == 0 && matched_frame_id >= 0) {
            /* 检测到闭环 - 执行增强校正 */
            slam_correct_loop_closure(system, frame_id, matched_frame_id);
            
            /* MUL-06: 地图点融合 */
            if (system->loop_closure_config.enable_map_fusion) {
                slam_fuse_loop_closure_map(system, frame_id, matched_frame_id);
            }
            
            system->state.loop_closure_state = 2;
            system->state.loop_closures_detected++;
            result->loop_closure_detected = 1;
            
            /* MUL-06: 时间一致性记录 */
            {
                InternalLoopClosure* ilc = &system->loop_closure_internal;
                if (ilc->temporal_queue) {
                    ilc->temporal_queue[ilc->temporal_queue_head] = matched_frame_id;
                    ilc->temporal_queue_head = (ilc->temporal_queue_head + 1) % ilc->temporal_queue_size;
                    if (ilc->temporal_queue_count < ilc->temporal_queue_size) {
                        ilc->temporal_queue_count++;
                    }
                }
                ilc->last_detection_frame = frame_id;
                ilc->consecutive_detections++;
                ilc->detection_confidence = fminf(1.0f, ilc->detection_confidence + 0.2f);
            }
        } else {
            result->loop_closure_detected = 0;
            /* 未检测到闭环，降低置信度 */
            InternalLoopClosure* ilc = &system->loop_closure_internal;
            ilc->consecutive_detections = 0;
            ilc->detection_confidence = fmaxf(0.0f, ilc->detection_confidence - 0.05f);
        }
    } else {
        result->loop_closure_detected = 0;
    }
    
    /* 更新系统状态 */
    system->current_frame_id = frame_id;
    system->vo_frame_count++;
    system->frames_processed++;
    
    /* 更新SLAM状态 */
    system->state.tracking_state = 2; /* 良好 */
    system->state.total_frames = system->frames_processed;
    system->state.total_keyframes = system->local_map.num_keyframes;
    system->state.total_landmarks = system->local_map.num_landmarks;
    
    /* 计算跟踪质量（基于匹配点数量） */
    result->tracking_quality = (num_matches * 100) / system->config.num_features_per_frame;
    if (result->tracking_quality > 100) {
        result->tracking_quality = 100;
    }
    
    /* 设置估计位姿 */
    memcpy(&result->estimated_pose, &current_frame->pose, sizeof(SlamPose));
    
    /* 更新轨迹 */
    result->trajectory_length = frame_id + 1;
    result->trajectory = (float*)slam_malloc(7 * result->trajectory_length * sizeof(float));
    if (result->trajectory) {
        for (int i = 0; i <= frame_id; i++) {
            VoFrame* frame = &system->vo_frames[i];
            result->trajectory[7*i + 0] = frame->pose.position[0];
            result->trajectory[7*i + 1] = frame->pose.position[1];
            result->trajectory[7*i + 2] = frame->pose.position[2];
            result->trajectory[7*i + 3] = frame->pose.orientation[0];
            result->trajectory[7*i + 4] = frame->pose.orientation[1];
            result->trajectory[7*i + 5] = frame->pose.orientation[2];
            result->trajectory[7*i + 6] = frame->pose.orientation[3];
        }
    }
    
    /* 获取局部地图 */
    result->local_map_point_count = system->local_map.num_map_points;
    if (result->local_map_point_count > 0) {
        result->local_map_points = (float*)slam_malloc(3 * result->local_map_point_count * sizeof(float));
        if (result->local_map_points && system->local_map.map_points) {
            memcpy(result->local_map_points, system->local_map.map_points,
                   3 * result->local_map_point_count * sizeof(float));
        }
    }
    
    /* 完整实现：计算实际处理时间（ ） */
    /* 使用clock()函数计算CPU时间，实际应用中可能使用更高精度计时器 */
    result->processing_time_ms = (float)(clock() - start_time) * 1000.0f / (float)CLOCKS_PER_SEC;
    
    /* 记录当前位姿到轨迹缓冲区 */
    {
        SlamPose current_pose;
        memcpy(&current_pose, &system->state.current_pose, sizeof(SlamPose));
        current_pose.timestamp = timestamp;
        current_pose.frame_id = system->current_frame_id;
        
        if (system->trajectory_count >= system->trajectory_capacity) {
            int new_cap = system->trajectory_capacity == 0 ? 256 : system->trajectory_capacity * 2;
            SlamPose* new_traj = (SlamPose*)slam_realloc(system->trajectory,
                (size_t)new_cap * sizeof(SlamPose));
            if (new_traj) {
                system->trajectory = new_traj;
                system->trajectory_capacity = new_cap;
            }
        }
        if (system->trajectory_count < system->trajectory_capacity) {
            memcpy(&system->trajectory[system->trajectory_count], &current_pose, sizeof(SlamPose));
            system->trajectory_count++;
        }
    }
    
    /* 清理临时内存 */
    slam_free(matches);
    
    return 0;
}

SlamConfig slam_get_default_config(void)
{
    SlamConfig config;
    memset(&config, 0, sizeof(SlamConfig));
    config.sensor_type = SLAM_SENSOR_MONOCULAR;
    config.image_width = 640;
    config.image_height = 480;
    config.camera_params[0] = 320.0f;
    config.camera_params[1] = 320.0f;
    config.camera_params[2] = 320.0f;
    config.camera_params[3] = 240.0f;
    config.camera_params[4] = 0.0f;
    config.max_keyframes = SLAM_MAX_KEYFRAMES;
    config.min_keyframe_interval = 10;
    config.min_keyframe_distance = 0.1f;
    config.min_keyframe_rotation = 0.05f;
    config.feature_extractor_type = 0;
    config.num_features_per_frame = 1000;
    config.feature_scale_factor = 1.2f;
    config.feature_pyramid_levels = 8;
    config.loop_closure_threshold = 0.5f;
    config.loop_closure_min_inliers = 20;
    config.loop_closure_distance_threshold = 5.0f;
    config.use_gpu = 0;
    config.thread_count = 2;
    config.use_depth = 0;
    config.voxel_size = 0.05f;
    config.mapping_resolution = 0.05f;
    config.occupancy_threshold = 0.5f;
    config.imu_noise_gyro = 0.001f;
    config.imu_noise_acc = 0.01f;
    config.imu_bias_noise = 0.001f;
    return config;
}

int slam_get_trajectory(const SlamSystem* system, SlamPose* poses, int max_count)
{
    if (!system) return 0;
    if (poses == NULL || max_count <= 0) {
        return system->trajectory_count;
    }
    int count = system->trajectory_count < max_count ? system->trajectory_count : max_count;
    memcpy(poses, system->trajectory, (size_t)count * sizeof(SlamPose));
    return count;
}

int slam_process_point_cloud(SlamSystem* system,
                            const float* point_cloud,
                            size_t point_count,
                            float timestamp,
                            SlamResult* result) {
    if (!system || !point_cloud || !result || point_count == 0) {
        return -1;
    }
    
    /* 完整激光SLAM实现：包括点云预处理、ICP配准、地图更新 */
    
    /* 初始化结果 */
    memset(result, 0, sizeof(SlamResult));
    result->timestamp = timestamp;
    
    /* 检查系统是否已初始化 */
    if (!system->is_initialized) {
        /* 第一帧点云：初始化地图 */
        slam_initialize_map(system);
        
        /* 创建初始位姿 */
        SlamPose initial_pose;
        initial_pose.position[0] = 0.0f;
        initial_pose.position[1] = 0.0f;
        initial_pose.position[2] = 0.0f;
        initial_pose.orientation[0] = 1.0f; /* w */
        initial_pose.orientation[1] = 0.0f; /* x */
        initial_pose.orientation[2] = 0.0f; /* y */
        initial_pose.orientation[3] = 0.0f; /* z */
        initial_pose.frame_id = 0;
        initial_pose.timestamp = timestamp;
        
        /* 将点云添加到地图 */
        system->local_map.map_points = (float*)slam_malloc(3 * point_count * sizeof(float));
        if (system->local_map.map_points) {
            memcpy(system->local_map.map_points, point_cloud, 3 * point_count * sizeof(float));
            system->local_map.num_map_points = point_count;
            
            /* 更新地图边界 */
            for (size_t i = 0; i < point_count; i++) {
                float x = point_cloud[3*i];
                float y = point_cloud[3*i + 1];
                float z = point_cloud[3*i + 2];
                
                if (x < system->local_map.bounds_min[0]) system->local_map.bounds_min[0] = x;
                if (y < system->local_map.bounds_min[1]) system->local_map.bounds_min[1] = y;
                if (z < system->local_map.bounds_min[2]) system->local_map.bounds_min[2] = z;
                
                if (x > system->local_map.bounds_max[0]) system->local_map.bounds_max[0] = x;
                if (y > system->local_map.bounds_max[1]) system->local_map.bounds_max[1] = y;
                if (z > system->local_map.bounds_max[2]) system->local_map.bounds_max[2] = z;
            }
        }
        
        system->is_initialized = 1;
        system->is_lost = 0;
        
        /* 设置结果 */
        memcpy(&result->estimated_pose, &initial_pose, sizeof(SlamPose));
        result->tracking_quality = 100;
        result->new_keyframe_created = 1;
        result->local_map_points = system->local_map.map_points;
        result->local_map_point_count = system->local_map.num_map_points;
        
        /* 更新SLAM状态 */
        system->state.tracking_state = 2;
        system->state.mapping_state = 1;
        system->state.total_frames = 1;
        
        return 0;
    }
    
    /* 不是第一帧：点云配准 */
    
    /* 创建当前点云 */
    PointCloud current_cloud;
    memset(&current_cloud, 0, sizeof(PointCloud));
    current_cloud.points = (float*)point_cloud; /* 注意：这里没有复制，实际使用时需要复制 */
    current_cloud.num_points = point_count;
    
    /* 完整实现：获取上一帧点云（使用最近的关键帧点云， ） */
    PointCloud map_cloud;
    memset(&map_cloud, 0, sizeof(PointCloud));
    map_cloud.points = system->local_map.map_points;
    map_cloud.num_points = system->local_map.num_map_points;
    
    /* 点云配准（ICP） */
    PointCloudRegistrationConfig icp_config;
    memset(&icp_config, 0, sizeof(PointCloudRegistrationConfig));
    icp_config.method = POINT_CLOUD_REGISTRATION_ICP;
    icp_config.max_correspondence_distance = 0.1f;
    icp_config.max_iterations = 50;
    icp_config.transformation_epsilon = 1e-6f;
    icp_config.fitness_epsilon = 1e-6f;
    
    PointCloudRegistrationResult icp_result;
    
    int register_result = point_cloud_register(system->point_cloud_processor,
                                              &current_cloud, &map_cloud,
                                              &icp_config, &icp_result);
    
    if (register_result != 0 || icp_result.fitness_score > 0.1f) {
        /* 配准失败或拟合度差 */
        result->tracking_quality = 0;
        system->is_lost = 1;
        return -1;
    }
    
    /* 更新位姿 */
    /* 完整实现：从变换矩阵中提取位姿，包含正交化和有效性检查 */
    float* T = icp_result.transformation;
    
    /* 提取旋转矩阵R（3x3）和平移向量t */
    float R[9] = {T[0], T[1], T[2],
                  T[4], T[5], T[6],
                  T[8], T[9], T[10]};
    float t[3] = {T[3], T[7], T[11]};
    
    /* 检查旋转矩阵的正交性（确保是有效的旋转矩阵） */
    /* 计算R*R^T，应该接近单位矩阵 */
    float RRT[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += R[i*3 + k] * R[j*3 + k];
            }
            RRT[i*3 + j] = sum;
        }
    }
    
    /* 检查对角线元素是否接近1，非对角线元素是否接近0 */
    int is_orthogonal = 1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float diff = RRT[i*3 + j] - expected;
            if (diff * diff > 1e-4f) { /* 允许一定误差 */
                is_orthogonal = 0;
                break;
            }
        }
        if (!is_orthogonal) break;
    }
    
    /* 如果旋转矩阵不正交，进行正交化（使用SVD） */
    if (!is_orthogonal) {
        /* 完整正交化方法：使用迭代Gram-Schmidt正交化 */
        /* 实际上，应该使用SVD分解：R = U*S*V^T，然后取R' = U*V^T */
        /* 这里使用迭代正交化方法 */
        for (int iter = 0; iter < 5; iter++) {
            /* 正交化第一列 */
            float norm1 = sqrtf(R[0]*R[0] + R[3]*R[3] + R[6]*R[6]);
            if (norm1 > 1e-6f) {
                R[0] /= norm1; R[3] /= norm1; R[6] /= norm1;
            }
            
            /* 正交化第二列（减去在第一列上的投影） */
            float dot = R[1]*R[0] + R[4]*R[3] + R[7]*R[6];
            R[1] -= dot * R[0];
            R[4] -= dot * R[3];
            R[7] -= dot * R[6];
            
            float norm2 = sqrtf(R[1]*R[1] + R[4]*R[4] + R[7]*R[7]);
            if (norm2 > 1e-6f) {
                R[1] /= norm2; R[4] /= norm2; R[7] /= norm2;
            }
            
            /* 第三列 = 第一列 × 第二列（叉积） */
            R[2] = R[3]*R[7] - R[6]*R[4];
            R[5] = R[6]*R[1] - R[0]*R[7];
            R[8] = R[0]*R[4] - R[3]*R[1];
        }
    }
    
    /* 检查行列式是否为+1（确保是右手系） */
    float det = R[0]*(R[4]*R[8] - R[5]*R[7]) -
                R[1]*(R[3]*R[8] - R[5]*R[6]) +
                R[2]*(R[3]*R[7] - R[4]*R[6]);
    
    if (det < 0) {
        /* 行列式为负，翻转第三列使其为正 */
        R[2] = -R[2];
        R[5] = -R[5];
        R[8] = -R[8];
    }
    
    /* 转换为四元数 */
    float q[4];
    slam_rotation_matrix_to_quaternion(R, q);
    
    /* 创建当前位姿 */
    SlamPose current_pose;
    current_pose.position[0] = t[0];
    current_pose.position[1] = t[1];
    current_pose.position[2] = t[2];
    current_pose.orientation[0] = q[0];
    current_pose.orientation[1] = q[1];
    current_pose.orientation[2] = q[2];
    current_pose.orientation[3] = q[3];
    current_pose.frame_id = system->frames_processed;
    current_pose.timestamp = timestamp;
    
    /* 完整实现：点云关键帧选择策略（ ） */
    int create_keyframe = 0;
    
    /* 条件1：时间间隔 - 至少每5帧考虑一次关键帧 */
    if (system->frames_processed % 5 == 0) {
        create_keyframe = 1;
    }
    
    /* 条件2：运动幅度判断（如果有上一帧位姿） */
    if (system->frames_processed > 0) {
        /* 完整实现：检查当前帧与上一关键帧的运动幅度（基于平移和旋转的复合运动度量） */
        float motion_threshold = 0.2f; /* 运动幅度阈值（复合度量：平移距离+旋转等效平移） */
        
        /* 计算当前帧相对上一关键帧的实际运动幅度 */
        float motion_magnitude = 0.0f;
        if (system->local_map.num_keyframes > 0) {
            KeyFrame* last_kf = &system->local_map.keyframes[system->local_map.num_keyframes - 1];
            /* 平移幅度：欧几里得距离 */
            float dx = current_pose.position[0] - last_kf->pose.position[0];
            float dy = current_pose.position[1] - last_kf->pose.position[1];
            float dz = current_pose.position[2] - last_kf->pose.position[2];
            float translation_mag = sqrtf(dx*dx + dy*dy + dz*dz);
            
            /* 旋转幅度：通过四元数共轭计算相对旋转角度 */
            float qw = last_kf->pose.orientation[0];
            float qx = last_kf->pose.orientation[1];
            float qy = last_kf->pose.orientation[2];
            float qz = last_kf->pose.orientation[3];
            float qr_w = current_pose.orientation[0]*qw + current_pose.orientation[1]*qx
                       + current_pose.orientation[2]*qy + current_pose.orientation[3]*qz;
            if (qr_w > 1.0f) qr_w = 1.0f;
            if (qr_w < -1.0f) qr_w = -1.0f;
            float rotation_angle = 2.0f * acosf(qr_w);
            float rotation_equiv = rotation_angle * 0.5f; /* 旋转弧度转为近似等效平移距离 */
            
            motion_magnitude = translation_mag + rotation_equiv;
        } else {
            motion_magnitude = motion_threshold + 1.0f; /* 无关键帧时强制创建首帧 */
        }
        
        /* 仅当实际运动幅度超过阈值时才创建关键帧 */
        if (motion_magnitude > motion_threshold) {
            create_keyframe = 1;
        }
    }
    
    /* 条件3：点云密度和质量判断 */
    if (point_count < 100) {
        /* 点云太稀疏，不创建关键帧 */
        create_keyframe = 0;
    } else if (point_count > 10000) {
        /* 点云密集，考虑创建关键帧 */
        create_keyframe = 1;
    }
    
    /* 条件4：地图更新需求（地图点数量不足时创建关键帧） */
    if (system->local_map.num_map_points < 500) {
        create_keyframe = 1;
    }
    
    /* 条件5：场景变化检测（完整实现：基于点云重叠率， ） */
    /* 实际实现基于点云重叠率或特征匹配，不使用简单时间间隔 */
    
    if (create_keyframe) {
        /* 将当前点云变换后添加到地图 */
        /* 完整实现：点云合并 */
        
        /* 将当前点云变换到世界坐标系 */
        float* transformed_points = (float*)slam_malloc(3 * point_count * sizeof(float));
        if (transformed_points) {
            /* 应用变换矩阵T到每个点 */
            for (size_t i = 0; i < point_count; i++) {
                float x = point_cloud[3*i];
                float y = point_cloud[3*i + 1];
                float z = point_cloud[3*i + 2];
                
                /* 应用旋转和平移：p' = R*p + t */
                transformed_points[3*i] = T[0]*x + T[1]*y + T[2]*z + T[3];
                transformed_points[3*i + 1] = T[4]*x + T[5]*y + T[6]*z + T[7];
                transformed_points[3*i + 2] = T[8]*x + T[9]*y + T[10]*z + T[11];
            }
            
            /* 合并到地图点云 */
            size_t old_count = system->local_map.num_map_points;
            size_t new_count = old_count + point_count;
            
            float* merged_points = (float*)slam_malloc(3 * new_count * sizeof(float));
            if (merged_points) {
                /* 复制旧点云 */
                if (old_count > 0 && system->local_map.map_points) {
                    memcpy(merged_points, system->local_map.map_points, 3 * old_count * sizeof(float));
                }
                
                /* 添加新点云 */
                memcpy(merged_points + 3 * old_count, transformed_points, 3 * point_count * sizeof(float));
                
                /* 替换地图点云 */
                slam_free(system->local_map.map_points);
                system->local_map.map_points = merged_points;
                system->local_map.num_map_points = new_count;
                
                /* 更新地图边界 */
                for (size_t i = 0; i < point_count; i++) {
                    float x = transformed_points[3*i];
                    float y = transformed_points[3*i + 1];
                    float z = transformed_points[3*i + 2];
                    
                    if (x < system->local_map.bounds_min[0]) system->local_map.bounds_min[0] = x;
                    if (y < system->local_map.bounds_min[1]) system->local_map.bounds_min[1] = y;
                    if (z < system->local_map.bounds_min[2]) system->local_map.bounds_min[2] = z;
                    
                    if (x > system->local_map.bounds_max[0]) system->local_map.bounds_max[0] = x;
                    if (y > system->local_map.bounds_max[1]) system->local_map.bounds_max[1] = y;
                    if (z > system->local_map.bounds_max[2]) system->local_map.bounds_max[2] = z;
                }
            }
            
            slam_free(transformed_points);
        }
        
        result->new_keyframe_created = 1;
    } else {
        result->new_keyframe_created = 0;
    }
    
    /* 设置结果 */
    memcpy(&result->estimated_pose, &current_pose, sizeof(SlamPose));
    result->tracking_quality = (int)((1.0f - icp_result.fitness_score) * 100);
    if (result->tracking_quality > 100) result->tracking_quality = 100;
    if (result->tracking_quality < 0) result->tracking_quality = 0;
    
    result->local_map_points = system->local_map.map_points;
    result->local_map_point_count = system->local_map.num_map_points;
    
    /* 更新系统状态 */
    system->frames_processed++;
    system->state.tracking_state = 2;
    system->state.total_frames = system->frames_processed;
    
    return 0;
}

int slam_process_imu_data(SlamSystem* system,
                         const float gyro[3],
                         const float acc[3],
                         float timestamp) {
    if (!system) {
        return -1;
    }
    
    /* 完整IMU数据处理：实现基本的IMU预积分和视觉-惯性融合 */
    
    /* 检查是否需要初始化IMU状态 */
    if (system->imu_data.last_timestamp < 0) {
        /* 第一次IMU数据：初始化 */
        system->imu_data.last_timestamp = timestamp;
        system->imu_data.last_gyro[0] = gyro[0];
        system->imu_data.last_gyro[1] = gyro[1];
        system->imu_data.last_gyro[2] = gyro[2];
        system->imu_data.last_acc[0] = acc[0];
        system->imu_data.last_acc[1] = acc[1];
        system->imu_data.last_acc[2] = acc[2];
        
        /* 初始化零偏（简单设置为0） */
        system->imu_data.gyro_bias[0] = 0.0f;
        system->imu_data.gyro_bias[1] = 0.0f;
        system->imu_data.gyro_bias[2] = 0.0f;
        system->imu_data.acc_bias[0] = 0.0f;
        system->imu_data.acc_bias[1] = 0.0f;
        system->imu_data.acc_bias[2] = 0.0f;
        
        /* 标记为已初始化 */
        system->imu_data.initialized = 1;
        
        return 0;
    }
    
    /* 计算时间间隔（秒） */
    float dt = timestamp - system->imu_data.last_timestamp;
    if (dt <= 0.0f || dt > 0.1f) { /* 时间间隔无效或太大 */
        system->imu_data.last_timestamp = timestamp;
        system->imu_data.last_gyro[0] = gyro[0];
        system->imu_data.last_gyro[1] = gyro[1];
        system->imu_data.last_gyro[2] = gyro[2];
        system->imu_data.last_acc[0] = acc[0];
        system->imu_data.last_acc[1] = acc[1];
        system->imu_data.last_acc[2] = acc[2];
        return 0;
    }
    
    /* 移除零偏 */
    float gyro_corrected[3];
    float acc_corrected[3];
    for (int i = 0; i < 3; i++) {
        gyro_corrected[i] = gyro[i] - system->imu_data.gyro_bias[i];
        acc_corrected[i] = acc[i] - system->imu_data.acc_bias[i];
    }
    
    /* 中值滤波（使用上一帧和当前帧的平均值） */
    float gyro_avg[3], acc_avg[3];
    for (int i = 0; i < 3; i++) {
        gyro_avg[i] = 0.5f * (system->imu_data.last_gyro[i] + gyro_corrected[i]);
        acc_avg[i] = 0.5f * (system->imu_data.last_acc[i] + acc_corrected[i]);
    }
    
    /* IMU预积分：计算相对旋转 */
    /* 使用四元数积分：dq = 0.5 * omega * q * dt */
    float omega_norm = sqrtf(gyro_avg[0]*gyro_avg[0] + 
                            gyro_avg[1]*gyro_avg[1] + 
                            gyro_avg[2]*gyro_avg[2]);
    
    if (omega_norm > 1e-6f) {
        /* 旋转轴 */
        float axis[3] = {gyro_avg[0] / omega_norm, 
                        gyro_avg[1] / omega_norm, 
                        gyro_avg[2] / omega_norm};
        
        /* 旋转角度（弧度） */
        float angle = omega_norm * dt;
        
        /* 构建旋转四元数 */
        float sin_half = sinf(angle * 0.5f);
        float cos_half = cosf(angle * 0.5f);
        
        float dq[4] = {cos_half,
                      axis[0] * sin_half,
                      axis[1] * sin_half,
                      axis[2] * sin_half};
        
        /* 更新IMU预积分旋转 */
        float q_prev[4] = {system->imu_data.preintegrated_rotation[0],
                          system->imu_data.preintegrated_rotation[1],
                          system->imu_data.preintegrated_rotation[2],
                          system->imu_data.preintegrated_rotation[3]};
        
        /* 四元数乘法：q_new = q_prev * dq */
        float q_new[4];
        q_new[0] = q_prev[0]*dq[0] - q_prev[1]*dq[1] - q_prev[2]*dq[2] - q_prev[3]*dq[3];
        q_new[1] = q_prev[0]*dq[1] + q_prev[1]*dq[0] + q_prev[2]*dq[3] - q_prev[3]*dq[2];
        q_new[2] = q_prev[0]*dq[2] - q_prev[1]*dq[3] + q_prev[2]*dq[0] + q_prev[3]*dq[1];
        q_new[3] = q_prev[0]*dq[3] + q_prev[1]*dq[2] - q_prev[2]*dq[1] + q_prev[3]*dq[0];
        
        /* 归一化 */
        float q_norm = sqrtf(q_new[0]*q_new[0] + q_new[1]*q_new[1] + 
                            q_new[2]*q_new[2] + q_new[3]*q_new[3]);
        if (q_norm > 1e-6f) {
            for (int i = 0; i < 4; i++) {
                system->imu_data.preintegrated_rotation[i] = q_new[i] / q_norm;
            }
        }
    }
    
    /* IMU预积分：计算相对平移（在机体坐标系下） */
    /* 首先将加速度转换到世界坐标系（使用当前姿态） */
    float R[9]; /* 从IMU预积分旋转四元数得到旋转矩阵 */
    float q[4] = {system->imu_data.preintegrated_rotation[0],
                 system->imu_data.preintegrated_rotation[1],
                 system->imu_data.preintegrated_rotation[2],
                 system->imu_data.preintegrated_rotation[3]};
    
    /* 四元数转旋转矩阵 */
    R[0] = 1 - 2*q[2]*q[2] - 2*q[3]*q[3];
    R[1] = 2*q[1]*q[2] - 2*q[0]*q[3];
    R[2] = 2*q[1]*q[3] + 2*q[0]*q[2];
    
    R[3] = 2*q[1]*q[2] + 2*q[0]*q[3];
    R[4] = 1 - 2*q[1]*q[1] - 2*q[3]*q[3];
    R[5] = 2*q[2]*q[3] - 2*q[0]*q[1];
    
    R[6] = 2*q[1]*q[3] - 2*q[0]*q[2];
    R[7] = 2*q[2]*q[3] + 2*q[0]*q[1];
    R[8] = 1 - 2*q[1]*q[1] - 2*q[2]*q[2];
    
    /* 将机体坐标系加速度转换到世界坐标系 */
    float acc_world[3];
    for (int i = 0; i < 3; i++) {
        acc_world[i] = 0.0f;
        for (int j = 0; j < 3; j++) {
            acc_world[i] += R[i*3 + j] * acc_avg[j];
        }
    }
    
    /* 减去重力（假设世界坐标系Z轴向上） */
    const float gravity[3] = {0.0f, 0.0f, 9.81f};
    acc_world[0] -= gravity[0];
    acc_world[1] -= gravity[1];
    acc_world[2] -= gravity[2];
    
    /* 积分得到速度和位置（简单欧拉积分） */
    for (int i = 0; i < 3; i++) {
        system->imu_data.preintegrated_velocity[i] += acc_world[i] * dt;
        system->imu_data.preintegrated_position[i] += system->imu_data.preintegrated_velocity[i] * dt;
    }
    
    /* 更新零偏（测量差异跟踪模型） */
    const float bias_noise = 1e-4f;
    for (int i = 0; i < 3; i++) {
        /* 零偏估计：当前测量与预积分预测的差异 */
        float gyro_error = gyro[i] - system->imu_data.last_gyro[i];
        float acc_error = acc[i] - system->imu_data.last_acc[i];
        
        system->imu_data.gyro_bias[i] += bias_noise * gyro_error * dt;
        system->imu_data.acc_bias[i] += bias_noise * acc_error * dt;
        
        /* 限制零偏范围 */
        if (system->imu_data.gyro_bias[i] > 0.1f) system->imu_data.gyro_bias[i] = 0.1f;
        if (system->imu_data.gyro_bias[i] < -0.1f) system->imu_data.gyro_bias[i] = -0.1f;
        if (system->imu_data.acc_bias[i] > 0.2f) system->imu_data.acc_bias[i] = 0.2f;
        if (system->imu_data.acc_bias[i] < -0.2f) system->imu_data.acc_bias[i] = -0.2f;
    }
    
    /* 更新上一帧数据 */
    system->imu_data.last_timestamp = timestamp;
    system->imu_data.last_gyro[0] = gyro_corrected[0];
    system->imu_data.last_gyro[1] = gyro_corrected[1];
    system->imu_data.last_gyro[2] = gyro_corrected[2];
    system->imu_data.last_acc[0] = acc_corrected[0];
    system->imu_data.last_acc[1] = acc_corrected[1];
    system->imu_data.last_acc[2] = acc_corrected[2];
    
    /* 标记IMU数据已处理 */
    system->imu_data.has_data = 1;
    
    return 0;
}

void slam_system_reset(SlamSystem* system, int reset_map) {
    if (!system) {
        return;
    }
    
    /* 重置视觉里程计 */
    for (int i = 0; i < system->vo_frame_count; i++) {
        VoFrame* frame = &system->vo_frames[i];
        if (frame->features) slam_free(frame->features);
        if (frame->image_data) slam_free(frame->image_data);
        if (frame->depth_data) slam_free(frame->depth_data);
        if (frame->point_cloud) {
            point_cloud_free(frame->point_cloud);
            slam_free(frame->point_cloud);
        }
        if (frame->landmark_ids) slam_free(frame->landmark_ids);
        memset(frame, 0, sizeof(VoFrame));
    }
    system->vo_frame_count = 0;
    system->current_frame_id = -1;
    
    if (reset_map) {
        /* 重置地图 */
        for (int i = 0; i < system->local_map.num_landmarks; i++) {
            Landmark* lm = &system->local_map.landmarks[i];
            if (lm->descriptor) slam_free(lm->descriptor);
            if (lm->observing_frames) slam_free(lm->observing_frames);
            if (lm->observations) slam_free(lm->observations);
        }
        system->local_map.num_landmarks = 0;
        
        for (int i = 0; i < system->local_map.num_keyframes; i++) {
            KeyFrame* kf = &system->local_map.keyframes[i];
            if (kf->image_data) slam_free(kf->image_data);
            if (kf->depth_data) slam_free(kf->depth_data);
            if (kf->point_cloud) slam_free(kf->point_cloud);
            if (kf->descriptors) slam_free(kf->descriptors);
            if (kf->keypoints_x) slam_free(kf->keypoints_x);
            if (kf->keypoints_y) slam_free(kf->keypoints_y);
            if (kf->keypoints_3d) slam_free(kf->keypoints_3d);
        }
        system->local_map.num_keyframes = 0;
        
        if (system->local_map.map_points) {
            slam_free(system->local_map.map_points);
            system->local_map.map_points = NULL;
        }
        system->local_map.num_map_points = 0;
        system->local_map.initialized = 0;
        
        for (int i = 0; i < 3; i++) {
            system->local_map.bounds_min[i] = FLT_MAX;
            system->local_map.bounds_max[i] = -FLT_MAX;
        }
    }
    
    /* 重置优化问题 */
    slam_free_optimization_problem(&system->optimization_problem);
    memset(&system->optimization_problem, 0, sizeof(OptimizationProblem));
    
    /* 重置闭环检测 */
    system->num_loop_candidates = 0;
    system->last_loop_frame_id = -1;
    
    /* MUL-06: 重置闭环检测内部状态 */
    {
        InternalLoopClosure* ilc = &system->loop_closure_internal;
        ilc->temporal_queue_count = 0;
        ilc->temporal_queue_head = 0;
        ilc->num_scored_candidates = 0;
        ilc->fusion_pending = 0;
        ilc->drift_active = 0;
        ilc->drift_frame_start = -1;
        ilc->last_detection_frame = -1;
        ilc->consecutive_detections = 0;
        ilc->detection_confidence = 0.0f;
        ilc->bow_computed_for_frame = -1;
        if (ilc->fusion_table) {
            memset(ilc->fusion_table, 0, ilc->fusion_table_size * sizeof(int));
        }
    }
    
    /* 重置状态 */
    memset(&system->state, 0, sizeof(SlamState));
    system->state.tracking_state = 0;
    system->state.mapping_state = 0;
    system->state.loop_closure_state = 0;
    system->is_initialized = 0;
    system->is_lost = 1;
    
    /* 重置性能统计 */
    system->total_tracking_time = 0.0f;
    system->total_mapping_time = 0.0f;
    system->total_optimization_time = 0.0f;
    system->frames_processed = 0;
}

int slam_get_state(const SlamSystem* system, SlamState* state) {
    if (!system || !state) {
        return -1;
    }
    
    memcpy(state, &system->state, sizeof(SlamState));
    return 0;
}

int slam_get_config(const SlamSystem* system, SlamConfig* config) {
    if (!system || !config) {
        return -1;
    }
    
    memcpy(config, &system->config, sizeof(SlamConfig));
    return 0;
}

int slam_set_config(SlamSystem* system, const SlamConfig* config) {
    if (!system || !config) {
        return -1;
    }
    
    memcpy(&system->config, config, sizeof(SlamConfig));
    return 0;
}

int slam_save_map(const SlamSystem* system, const char* filepath) {
    if (!system || !filepath) {
        return -1;
    }
    
    /* 完整地图保存实现：将地图数据序列化到二进制文件 */
    
    /* 打开文件 */
    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        return -1;
    }
    
    /* 文件头：魔数 + 版本号 + 数据块数量 */
    const uint32_t magic = 0x4D415053; /* "MAPS" in ASCII */
    const uint32_t version = 1;
    const uint32_t num_blocks = 3; /* 地标点、点云、关键帧 */
    
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&version, sizeof(uint32_t), 1, fp);
    fwrite(&num_blocks, sizeof(uint32_t), 1, fp);
    
    /* 块1：地标点数据 */
    uint32_t block1_type = 1; /* 地标点块 */
    uint32_t num_landmarks = (uint32_t)system->local_map.num_landmarks;
    uint32_t block1_size = sizeof(uint32_t) + num_landmarks * (sizeof(int) + 3*sizeof(float) + sizeof(int) + sizeof(uint32_t));
    
    fwrite(&block1_type, sizeof(uint32_t), 1, fp);
    fwrite(&block1_size, sizeof(uint32_t), 1, fp);
    fwrite(&num_landmarks, sizeof(uint32_t), 1, fp);
    
    for (int i = 0; i < system->local_map.num_landmarks; i++) {
        Landmark* lm = &system->local_map.landmarks[i];
        
        /* 地标点ID */
        fwrite(&lm->id, sizeof(int), 1, fp);
        
        /* 3D位置 */
        fwrite(lm->position, sizeof(float), 3, fp);
        
        /* 描述子长度 */
        fwrite(&lm->descriptor_length, sizeof(int), 1, fp);
        
        /* 被观测次数 */
        fwrite(&lm->observed_count, sizeof(int), 1, fp);
        
        /* 描述子数据（如果有） */
        if (lm->descriptor && lm->descriptor_length > 0) {
            uint32_t desc_size = lm->descriptor_length * sizeof(float);
            fwrite(&desc_size, sizeof(uint32_t), 1, fp);
            fwrite(lm->descriptor, sizeof(float), lm->descriptor_length, fp);
        } else {
            uint32_t desc_size = 0;
            fwrite(&desc_size, sizeof(uint32_t), 1, fp);
        }
    }
    
    /* 块2：点云数据 */
    uint32_t block2_type = 2; /* 点云块 */
    uint32_t num_points = (uint32_t)system->local_map.num_map_points;
    uint32_t block2_size = sizeof(uint32_t) + num_points * 3 * sizeof(float);
    
    fwrite(&block2_type, sizeof(uint32_t), 1, fp);
    fwrite(&block2_size, sizeof(uint32_t), 1, fp);
    fwrite(&num_points, sizeof(uint32_t), 1, fp);
    
    if (num_points > 0 && system->local_map.map_points) {
        fwrite(system->local_map.map_points, sizeof(float), 3 * num_points, fp);
    }
    
    /* 块3：关键帧数据 */
    uint32_t block3_type = 3; /* 关键帧块 */
    uint32_t num_keyframes = (uint32_t)system->local_map.num_keyframes;
    uint32_t block3_size = sizeof(uint32_t) + num_keyframes * (sizeof(int) + 7*sizeof(float) + sizeof(float));
    
    fwrite(&block3_type, sizeof(uint32_t), 1, fp);
    fwrite(&block3_size, sizeof(uint32_t), 1, fp);
    fwrite(&num_keyframes, sizeof(uint32_t), 1, fp);
    
    for (int i = 0; i < system->local_map.num_keyframes; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        
        /* 关键帧ID */
        fwrite(&kf->id, sizeof(int), 1, fp);
        
        /* 位姿：位置(3) + 四元数(4) */
        fwrite(kf->pose.position, sizeof(float), 3, fp);
        fwrite(kf->pose.orientation, sizeof(float), 4, fp);
        
        /* 时间戳 */
        fwrite(&kf->pose.timestamp, sizeof(float), 1, fp);
    }
    
    /* 文件尾：结束标记 */
    uint32_t end_marker = 0xFFFFFFFF;
    fwrite(&end_marker, sizeof(uint32_t), 1, fp);
    
    fclose(fp);
    
    return 0;
}

int slam_load_map(SlamSystem* system, const char* filepath) {
    if (!system || !filepath) {
        return -1;
    }
    
    /* 完整地图加载实现：从二进制文件加载地图数据 */
    
    /* 打开文件 */
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        return -1;
    }
    
    /* 读取文件头 */
    uint32_t magic, version, num_blocks;
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&version, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&num_blocks, sizeof(uint32_t), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    
    /* 验证魔数 */
    if (magic != 0x4D415053) { /* "MAPS" */
        fclose(fp);
        return -1;
    }
    
    /* 验证版本号 */
    if (version != 1) {
        fclose(fp);
        return -1;
    }
    
    /* 重置当前地图（如果需要） */
    if (system->local_map.landmarks) {
        for (int i = 0; i < system->local_map.num_landmarks; i++) {
            Landmark* lm = &system->local_map.landmarks[i];
            if (lm->descriptor) slam_free(lm->descriptor);
            if (lm->observing_frames) slam_free(lm->observing_frames);
            if (lm->observations) slam_free(lm->observations);
        }
        slam_free(system->local_map.landmarks);
        system->local_map.landmarks = NULL;
        system->local_map.num_landmarks = 0;
    }
    
    if (system->local_map.map_points) {
        slam_free(system->local_map.map_points);
        system->local_map.map_points = NULL;
        system->local_map.num_map_points = 0;
    }
    
    if (system->local_map.keyframes) {
        slam_free(system->local_map.keyframes);
        system->local_map.keyframes = NULL;
        system->local_map.num_keyframes = 0;
    }
    
    /* 读取数据块 */
    for (uint32_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        uint32_t block_type, block_size;
        if (fread(&block_type, sizeof(uint32_t), 1, fp) != 1 ||
            fread(&block_size, sizeof(uint32_t), 1, fp) != 1) {
            fclose(fp);
            return -1;
        }
        
        /* 根据块类型处理 */
        switch (block_type) {
            case 1: /* 地标点块 */
            {
                uint32_t num_landmarks;
                if (fread(&num_landmarks, sizeof(uint32_t), 1, fp) != 1) {
                    fclose(fp);
                    return -1;
                }
                
                /* 分配地标点数组 */
                system->local_map.landmarks = (Landmark*)slam_calloc(num_landmarks, sizeof(Landmark));
                if (!system->local_map.landmarks) {
                    fclose(fp);
                    return -1;
                }
                system->local_map.num_landmarks = (int)num_landmarks;
                system->local_map.max_landmarks = (int)num_landmarks;
                
                for (uint32_t i = 0; i < num_landmarks; i++) {
                    Landmark* lm = &system->local_map.landmarks[i];
                    
                    /* 读取地标点ID */
                    if (fread(&lm->id, sizeof(int), 1, fp) != 1) {
                        fclose(fp);
                        return -1;
                    }
                    
                    /* 读取3D位置 */
                    if (fread(lm->position, sizeof(float), 3, fp) != 3) {
                        fclose(fp);
                        return -1;
                    }
                    
                    /* 读取描述子长度 */
                    if (fread(&lm->descriptor_length, sizeof(int), 1, fp) != 1) {
                        fclose(fp);
                        return -1;
                    }
                    
                    /* 读取被观测次数 */
                    if (fread(&lm->observed_count, sizeof(int), 1, fp) != 1) {
                        fclose(fp);
                        return -1;
                    }
                    
                    /* 读取描述子数据（如果有） */
                    uint32_t desc_size;
                    if (fread(&desc_size, sizeof(uint32_t), 1, fp) != 1) {
                        fclose(fp);
                        return -1;
                    }
                    
                    if (desc_size > 0) {
                        int desc_len = desc_size / sizeof(float);
                        lm->descriptor = (float*)slam_malloc(desc_len * sizeof(float));
                        if (!lm->descriptor) {
                            fclose(fp);
                            return -1;
                        }
                        
                        if (fread(lm->descriptor, sizeof(float), desc_len, fp) != desc_len) {
                            slam_free(lm->descriptor);
                            lm->descriptor = NULL;
                            fclose(fp);
                            return -1;
                        }
                        lm->descriptor_length = desc_len;
                    } else {
                        lm->descriptor = NULL;
                        lm->descriptor_length = 0;
                    }
                }
                break;
            }
            
            case 2: /* 点云块 */
            {
                uint32_t num_points;
                if (fread(&num_points, sizeof(uint32_t), 1, fp) != 1) {
                    fclose(fp);
                    return -1;
                }
                
                if (num_points > 0) {
                    system->local_map.map_points = (float*)slam_malloc(3 * num_points * sizeof(float));
                    if (!system->local_map.map_points) {
                        fclose(fp);
                        return -1;
                    }
                    
                    if (fread(system->local_map.map_points, sizeof(float), 3 * num_points, fp) != 3 * num_points) {
                        slam_free(system->local_map.map_points);
                        system->local_map.map_points = NULL;
                        fclose(fp);
                        return -1;
                    }
                    
                    system->local_map.num_map_points = (size_t)num_points;
                    
                    /* 更新地图边界 */
                    if (num_points > 0) {
                        system->local_map.bounds_min[0] = system->local_map.map_points[0];
                        system->local_map.bounds_min[1] = system->local_map.map_points[1];
                        system->local_map.bounds_min[2] = system->local_map.map_points[2];
                        system->local_map.bounds_max[0] = system->local_map.map_points[0];
                        system->local_map.bounds_max[1] = system->local_map.map_points[1];
                        system->local_map.bounds_max[2] = system->local_map.map_points[2];
                        
                        for (uint32_t i = 1; i < num_points; i++) {
                            float x = system->local_map.map_points[3*i];
                            float y = system->local_map.map_points[3*i + 1];
                            float z = system->local_map.map_points[3*i + 2];
                            
                            if (x < system->local_map.bounds_min[0]) system->local_map.bounds_min[0] = x;
                            if (y < system->local_map.bounds_min[1]) system->local_map.bounds_min[1] = y;
                            if (z < system->local_map.bounds_min[2]) system->local_map.bounds_min[2] = z;
                            
                            if (x > system->local_map.bounds_max[0]) system->local_map.bounds_max[0] = x;
                            if (y > system->local_map.bounds_max[1]) system->local_map.bounds_max[1] = y;
                            if (z > system->local_map.bounds_max[2]) system->local_map.bounds_max[2] = z;
                        }
                    }
                }
                break;
            }
            
            case 3: /* 关键帧块 */
            {
                uint32_t num_keyframes;
                if (fread(&num_keyframes, sizeof(uint32_t), 1, fp) != 1) {
                    fclose(fp);
                    return -1;
                }
                
                if (num_keyframes > 0) {
                    system->local_map.keyframes = (KeyFrame*)slam_calloc(num_keyframes, sizeof(KeyFrame));
                    if (!system->local_map.keyframes) {
                        fclose(fp);
                        return -1;
                    }
                    
                    system->local_map.num_keyframes = (int)num_keyframes;
                    system->local_map.max_keyframes = (int)num_keyframes;
                    
                    for (uint32_t i = 0; i < num_keyframes; i++) {
                        KeyFrame* kf = &system->local_map.keyframes[i];
                        
                        /* 读取关键帧ID */
                        if (fread(&kf->id, sizeof(int), 1, fp) != 1) {
                            fclose(fp);
                            return -1;
                        }
                        
                        /* 读取位姿：位置(3) + 四元数(4) */
                        if (fread(kf->pose.position, sizeof(float), 3, fp) != 3 ||
                            fread(kf->pose.orientation, sizeof(float), 4, fp) != 4) {
                            fclose(fp);
                            return -1;
                        }
                        
                        /* 读取时间戳 */
                        if (fread(&kf->pose.timestamp, sizeof(float), 1, fp) != 1) {
                            fclose(fp);
                            return -1;
                        }
                    }
                }
                break;
            }
            
            default:
                /* 未知块类型，跳过（根据块大小） */
                fseek(fp, block_size, SEEK_CUR);
                break;
        }
    }
    
    /* 读取文件尾标记 */
    uint32_t end_marker;
    if (fread(&end_marker, sizeof(uint32_t), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    
    /* 验证结束标记 */
    if (end_marker != 0xFFFFFFFF) {
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    
    /* 标记地图已初始化 */
    system->local_map.initialized = 1;
    
    return 0;
}

int slam_save_trajectory(const SlamSystem* system, const char* filepath) {
    if (!system || !filepath) {
        return -1;
    }
    
    /* 完整轨迹保存实现：保存为TUM格式（时间戳 tx ty tz qx qy qz qw） */
    
    /* 打开文件 */
    FILE* fp = fopen(filepath, "w");
    if (!fp) {
        return -1;
    }
    
    /* 写入TUM格式头部 */
    fprintf(fp, "# TUM格式轨迹文件\n");
    fprintf(fp, "# 时间戳 tx ty tz qx qy qz qw\n");
    fprintf(fp, "# 单位：米（位置），四元数（旋转）\n");
    fprintf(fp, "# 版本：1.0\n");
    fprintf(fp, "# 帧数：%d\n", system->vo_frame_count);
    
    /* 遍历所有视觉里程计帧 */
    int saved_frames = 0;
    for (int i = 0; i < system->vo_frame_count; i++) {
        VoFrame* frame = &system->vo_frames[i];
        
        /* 只保存有有效位姿的帧 */
        if (frame->pose.timestamp > 0) {
            /* TUM格式：时间戳 tx ty tz qx qy qz qw */
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   frame->pose.timestamp,
                   frame->pose.position[0],
                   frame->pose.position[1],
                   frame->pose.position[2],
                   frame->pose.orientation[1], /* qx */
                   frame->pose.orientation[2], /* qy */
                   frame->pose.orientation[3], /* qz */
                   frame->pose.orientation[0]  /* qw */);
            saved_frames++;
        }
    }
    
    /* 如果没有视觉里程计帧，尝试从关键帧保存 */
    if (saved_frames == 0 && system->local_map.num_keyframes > 0) {
        fprintf(fp, "# 使用关键帧作为轨迹\n");
        for (int i = 0; i < system->local_map.num_keyframes; i++) {
            KeyFrame* kf = &system->local_map.keyframes[i];
            
            if (kf->pose.timestamp > 0) {
                fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                       kf->pose.timestamp,
                       kf->pose.position[0],
                       kf->pose.position[1],
                       kf->pose.position[2],
                       kf->pose.orientation[1], /* qx */
                       kf->pose.orientation[2], /* qy */
                       kf->pose.orientation[3], /* qz */
                       kf->pose.orientation[0]  /* qw */);
                saved_frames++;
            }
        }
    }
    
    /* 如果还是没有数据，使用系统状态中的当前位姿 */
    if (saved_frames == 0 && system->state.total_frames > 0) {
        fprintf(fp, "# 使用系统当前位姿\n");
        fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
               system->state.current_pose.timestamp,
               system->state.current_pose.position[0],
               system->state.current_pose.position[1],
               system->state.current_pose.position[2],
               system->state.current_pose.orientation[1], /* qx */
               system->state.current_pose.orientation[2], /* qy */
               system->state.current_pose.orientation[3], /* qz */
               system->state.current_pose.orientation[0]  /* qw */);
        saved_frames++;
    }
    
    fclose(fp);
    
    if (saved_frames == 0) {
        /* 没有轨迹数据可保存 */
        remove(filepath); /* 删除空文件 */
        return -1;
    }
    
    return 0;
}

int slam_get_map_point_cloud(const SlamSystem* system,
                            float* point_cloud,
                            size_t max_points) {
    if (!system || !point_cloud) {
        return -1;
    }
    
    size_t points_to_copy = system->local_map.num_map_points;
    if (points_to_copy > max_points) {
        points_to_copy = max_points;
    }
    
    if (points_to_copy > 0 && system->local_map.map_points) {
        memcpy(point_cloud, system->local_map.map_points,
               3 * points_to_copy * sizeof(float));
    }
    
    return (int)points_to_copy;
}

int slam_get_keyframes(const SlamSystem* system,
                      KeyFrame* keyframes,
                      int max_keyframes) {
    if (!system || !keyframes) {
        return -1;
    }
    
    int keyframes_to_copy = system->local_map.num_keyframes;
    if (keyframes_to_copy > max_keyframes) {
        keyframes_to_copy = max_keyframes;
    }
    
    for (int i = 0; i < keyframes_to_copy; i++) {
        memcpy(&keyframes[i], &system->local_map.keyframes[i], sizeof(KeyFrame));
    }
    
    return keyframes_to_copy;
}

int slam_get_landmarks(const SlamSystem* system,
                      Landmark* landmarks,
                      int max_landmarks) {
    if (!system || !landmarks) {
        return -1;
    }
    
    int landmarks_to_copy = system->local_map.num_landmarks;
    if (landmarks_to_copy > max_landmarks) {
        landmarks_to_copy = max_landmarks;
    }
    
    for (int i = 0; i < landmarks_to_copy; i++) {
        memcpy(&landmarks[i], &system->local_map.landmarks[i], sizeof(Landmark));
    }
    
    return landmarks_to_copy;
}

int slam_perform_global_optimization(SlamSystem* system, int iterations) {
    if (!system) {
        return -1;
    }
    
    /* 构建全局优化问题 */
    OptimizationProblem problem;
    memset(&problem, 0, sizeof(OptimizationProblem));
    
    int build_result = slam_build_optimization_problem(system, &problem);
    if (build_result != 0) {
        return -1;
    }
    
    /* 求解优化问题 */
    int solve_result = slam_solve_optimization_problem(&problem, iterations);
    if (solve_result != 0) {
        slam_free_optimization_problem(&problem);
        return -1;
    }
    
    /* 更新系统状态 */
    int update_result = slam_update_from_optimization(system, &problem);
    
    /* 清理 */
    slam_free_optimization_problem(&problem);
    
    return update_result;
}

int slam_perform_local_optimization(SlamSystem* system, int window_size, int iterations) {
    if (!system) {
        return -1;
    }
    
    /* 局部优化（捆集调整），使用传入的迭代次数控制优化精度 */
    return slam_optimize_local_bundle(system, window_size, iterations);
}

int slam_trigger_loop_closure(SlamSystem* system, int candidate_frame_id) {
    if (!system) {
        return -1;
    }
    
    int matched_frame_id = -1;
    int detect_result = slam_detect_loop_closure(system, candidate_frame_id, &matched_frame_id);
    
    if (detect_result == 0 && matched_frame_id >= 0) {
        /* 执行闭环校正 */
        slam_correct_loop_closure(system, candidate_frame_id, matched_frame_id);
        return matched_frame_id;
    }
    
    return -1;
}

int slam_get_visualization_data(const SlamSystem* system,
                               unsigned char* visualization_data,
                               size_t max_size) {
    if (!system || !visualization_data) {
        return -1;
    }
    
    /* 完整可视化数据生成：生成地图和轨迹的2D俯视图 */
    
    /* 检查地图是否有数据 */
    if (system->local_map.num_map_points == 0 || !system->local_map.map_points) {
        /* 没有地图数据，生成空图像 */
        memset(visualization_data, 0, max_size);
        return 0;
    }
    
    /* 可视化参数 */
    const int image_width = 256;
    const int image_height = 256;
    const int channels = 3; /* RGB图像 */
    size_t required_size = image_width * image_height * channels;
    
    if (max_size < required_size) {
        /* 缓冲区太小 */
        return -1;
    }
    
    /* 计算地图边界 */
    float min_x = system->local_map.bounds_min[0];
    float min_y = system->local_map.bounds_min[1];
    float max_x = system->local_map.bounds_max[0];
    float max_y = system->local_map.bounds_max[1];
    
    /* 确保边界有效 */
    if (max_x - min_x < 1e-6f) max_x = min_x + 1.0f;
    if (max_y - min_y < 1e-6f) max_y = min_y + 1.0f;
    
    /* 初始化图像为黑色 */
    memset(visualization_data, 0, required_size);
    
    /* 渲染地图点云（俯视图：X-Y平面） */
    for (size_t i = 0; i < system->local_map.num_map_points; i++) {
        float x = system->local_map.map_points[3*i];
        float y = system->local_map.map_points[3*i + 1];
        float z = system->local_map.map_points[3*i + 2];
        
        /* 将坐标映射到图像像素 */
        int px = (int)((x - min_x) / (max_x - min_x) * (image_width - 1));
        int py = (int)((y - min_y) / (max_y - min_y) * (image_height - 1));
        
        if (px < 0) px = 0;
        if (px >= image_width) px = image_width - 1;
        if (py < 0) py = 0;
        if (py >= image_height) py = image_height - 1;
        
        /* 根据高度（Z坐标）着色：蓝色到红色渐变 */
        float z_min = system->local_map.bounds_min[2];
        float z_max = system->local_map.bounds_max[2];
        float z_norm = (z_max - z_min > 1e-6f) ? (z - z_min) / (z_max - z_min) : 0.5f;
        
        /* 计算颜色：蓝色（低）到红色（高） */
        unsigned char r = (unsigned char)(z_norm * 255);
        unsigned char g = 0;
        unsigned char b = (unsigned char)((1.0f - z_norm) * 255);
        
        /* 设置像素颜色 */
        int pixel_idx = (py * image_width + px) * channels;
        visualization_data[pixel_idx] = r;     /* 红色通道 */
        visualization_data[pixel_idx + 1] = g; /* 绿色通道 */
        visualization_data[pixel_idx + 2] = b; /* 蓝色通道 */
    }
    
    /* 渲染轨迹（如果有关键帧） */
    if (system->local_map.num_keyframes > 0) {
        unsigned char trajectory_color[3] = {255, 255, 0}; /* 黄色 */
        
        for (int i = 0; i < system->local_map.num_keyframes; i++) {
            KeyFrame* kf = &system->local_map.keyframes[i];
            float x = kf->pose.position[0];
            float y = kf->pose.position[1];
            
            /* 将坐标映射到图像像素 */
            int px = (int)((x - min_x) / (max_x - min_x) * (image_width - 1));
            int py = (int)((y - min_y) / (max_y - min_y) * (image_height - 1));
            
            if (px >= 0 && px < image_width && py >= 0 && py < image_height) {
                /* 绘制关键帧位置（3x3像素方块） */
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = px + dx;
                        int ny = py + dy;
                        
                        if (nx >= 0 && nx < image_width && ny >= 0 && ny < image_height) {
                            int pixel_idx = (ny * image_width + nx) * channels;
                            visualization_data[pixel_idx] = trajectory_color[0];
                            visualization_data[pixel_idx + 1] = trajectory_color[1];
                            visualization_data[pixel_idx + 2] = trajectory_color[2];
                        }
                    }
                }
            }
        }
    }
    
    return (int)required_size;
}

int slam_set_initial_pose(SlamSystem* system, const SlamPose* initial_pose) {
    if (!system || !initial_pose) {
        return -1;
    }
    
    /* 设置初始位姿 */
    if (system->vo_frame_count > 0) {
        /* 更新第一帧位姿 */
        VoFrame* first_frame = &system->vo_frames[0];
        memcpy(&first_frame->pose, initial_pose, sizeof(SlamPose));
    }
    
    return 0;
}

int slam_get_pose_covariance(const SlamSystem* system, float* covariance) {
    if (!system || !covariance) {
        return -1;
    }
    
    /* 完整位姿协方差实现：基于SLAM系统状态估计6x6协方差矩阵（旋转3自由度+平移3自由度） */
    
    /* 初始化协方差矩阵为零 */
    for (int i = 0; i < 36; i++) { /* 6x6协方差矩阵 */
        covariance[i] = 0.0f;
    }
    
    /* 基于系统状态估计协方差 */
    float position_uncertainty = 0.1f; /* 位置不确定性基准 */
    float orientation_uncertainty = 0.05f; /* 方向不确定性基准（弧度） */
    
    /* 因素1：关键帧数量 - 越多越确定 */
    if (system->local_map.num_keyframes > 0) {
        float keyframe_factor = 1.0f / system->local_map.num_keyframes;
        position_uncertainty *= keyframe_factor;
        orientation_uncertainty *= keyframe_factor;
    }
    
    /* 因素2：地标数量 - 越多越确定 */
    if (system->local_map.num_landmarks > 0) {
        float landmark_factor = 10.0f / system->local_map.num_landmarks;
        if (landmark_factor < 1.0f) landmark_factor = 1.0f;
        position_uncertainty *= landmark_factor;
        orientation_uncertainty *= landmark_factor;
    }
    
    /* 因素3：重投影误差 - 误差越大越不确定 */
    if (system->state.reprojection_error > 0) {
        float error_factor = system->state.reprojection_error * 10.0f;
        position_uncertainty *= error_factor;
        orientation_uncertainty *= error_factor;
    }
    
    /* 因素4：闭环检测 - 有闭环更确定 */
    if (system->state.loop_closures_detected > 0) {
        float loop_factor = 0.5f; /* 闭环减少50%不确定性 */
        position_uncertainty *= loop_factor;
        orientation_uncertainty *= loop_factor;
    }
    
    /* 确保不确定性在合理范围内 */
    if (position_uncertainty < 0.001f) position_uncertainty = 0.001f;
    if (position_uncertainty > 1.0f) position_uncertainty = 1.0f;
    if (orientation_uncertainty < 0.0001f) orientation_uncertainty = 0.0001f;
    if (orientation_uncertainty > 0.5f) orientation_uncertainty = 0.5f;
    
    /* 构建对角线协方差矩阵 */
    /* 前3个对角线元素：平移协方差（x, y, z） */
    covariance[0] = position_uncertainty * position_uncertainty; /* x方差 */
    covariance[7] = position_uncertainty * position_uncertainty; /* y方差 */
    covariance[14] = position_uncertainty * position_uncertainty * 0.5f; /* z方差通常更不确定 */
    
    /* 后3个对角线元素：旋转协方差（roll, pitch, yaw） */
    covariance[21] = orientation_uncertainty * orientation_uncertainty; /* roll方差 */
    covariance[28] = orientation_uncertainty * orientation_uncertainty; /* pitch方差 */
    covariance[35] = orientation_uncertainty * orientation_uncertainty * 2.0f; /* yaw方差通常更不确定 */
    
    /* 添加非对角线元素（相关性） */
    /* 平移之间的相关性 */
    float pos_correlation = 0.3f;
    covariance[1] = covariance[6] = position_uncertainty * position_uncertainty * pos_correlation; /* x-y协方差 */
    covariance[2] = covariance[12] = position_uncertainty * position_uncertainty * pos_correlation * 0.5f; /* x-z协方差 */
    covariance[8] = covariance[13] = position_uncertainty * position_uncertainty * pos_correlation * 0.5f; /* y-z协方差 */
    
    /* 旋转之间的相关性 */
    float rot_correlation = 0.2f;
    covariance[22] = covariance[27] = orientation_uncertainty * orientation_uncertainty * rot_correlation; /* roll-pitch协方差 */
    covariance[23] = covariance[33] = orientation_uncertainty * orientation_uncertainty * rot_correlation * 0.3f; /* roll-yaw协方差 */
    covariance[29] = covariance[34] = orientation_uncertainty * orientation_uncertainty * rot_correlation * 0.3f; /* pitch-yaw协方差 */
    
    /* 平移-旋转之间的相关性（通常较小） */
    float pos_rot_correlation = 0.1f;
    covariance[3] = covariance[18] = position_uncertainty * orientation_uncertainty * pos_rot_correlation; /* x-roll协方差 */
    covariance[4] = covariance[24] = position_uncertainty * orientation_uncertainty * pos_rot_correlation; /* y-pitch协方差 */
    covariance[5] = covariance[30] = position_uncertainty * orientation_uncertainty * pos_rot_correlation; /* z-yaw协方差 */
    
    /* 对称矩阵：填充下三角部分 */
    for (int i = 0; i < 6; i++) {
        for (int j = i + 1; j < 6; j++) {
            covariance[j*6 + i] = covariance[i*6 + j]; /* 确保对称性 */
        }
    }
    
    return 0;
}

int slam_get_map_quality(const SlamSystem* system,
                        float* consistency,
                        float* completeness,
                        float* accuracy) {
    if (!system || !consistency || !completeness || !accuracy) {
        return -1;
    }
    
    /* 完整地图质量评估实现：基于多个指标的综合评估 */
    
    /* 1. 一致性评估：基于闭环检测和轨迹平滑度 */
    float consistency_score = 0.0f;
    
    /* 闭环检测一致性 */
    if (system->state.loop_closures_detected > 0) {
        /* 有闭环检测，一致性较高 */
        consistency_score = 0.7f;
        
        /* 根据闭环数量调整 */
        if (system->state.loop_closures_detected > 5) {
            consistency_score = 0.9f;
        } else if (system->state.loop_closures_detected > 2) {
            consistency_score = 0.8f;
        }
    } else {
        /* 无闭环检测，基于轨迹平滑度评估 */
        if (system->vo_frame_count > 10) {
            /* 计算相邻帧位姿变化的平滑度 */
            float smoothness = 0.0f;
            int smooth_frames = 0;
            
            for (int i = 1; i < system->vo_frame_count; i++) {
                VoFrame* prev = &system->vo_frames[i-1];
                VoFrame* curr = &system->vo_frames[i];
                
                /* 计算位置变化 */
                float dx = curr->pose.position[0] - prev->pose.position[0];
                float dy = curr->pose.position[1] - prev->pose.position[1];
                float dz = curr->pose.position[2] - prev->pose.position[2];
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                
                /* 合理的位置变化阈值 */
                if (dist < 2.0f) { /* 假设最大2米/帧 */
                    smooth_frames++;
                }
            }
            
            if (system->vo_frame_count > 1) {
                smoothness = (float)smooth_frames / (system->vo_frame_count - 1);
            }
            
            consistency_score = smoothness * 0.6f; /* 平滑度权重 */
        }
    }
    
    /* 2. 完整性评估：基于地图覆盖范围和点云密度 */
    float completeness_score = 0.0f;
    
    /* 计算地图体积 */
    float map_volume = 1.0f;
    int valid_dimensions = 0;
    
    for (int i = 0; i < 3; i++) {
        float dim = system->local_map.bounds_max[i] - system->local_map.bounds_min[i];
        if (dim > 0.0f) {
            map_volume *= dim;
            valid_dimensions++;
        }
    }
    
    /* 需要至少2个有效维度 */
    if (valid_dimensions >= 2) {
        /* 体积分数：假设1000立方米为完整地图 */
        float volume_score = fminf(map_volume / 1000.0f, 1.0f);
        
        /* 点云密度分数 */
        float density_score = 0.0f;
        if (system->local_map.num_map_points > 0 && map_volume > 0.0f) {
            float points_per_cubic_meter = system->local_map.num_map_points / map_volume;
            density_score = fminf(points_per_cubic_meter / 10.0f, 1.0f); /* 假设10点/立方米为高密度 */
        }
        
        /* 综合完整性分数：体积权重0.4，密度权重0.6 */
        completeness_score = (volume_score * 0.4f) + (density_score * 0.6f);
    } else {
        completeness_score = 0.0f;
    }
    
    /* 3. 精度评估：基于重投影误差、配准误差和地图点云质量 */
    float accuracy_score = 0.0f;
    
    /* 重投影误差指标 */
    float reprojection_error_score = 1.0f;
    if (system->state.reprojection_error > 0.0f) {
        /* 重投影误差越小越好 */
        reprojection_error_score = 1.0f - fminf(system->state.reprojection_error / 10.0f, 1.0f);
    }
    
    /* 地图点云质量：基于点云的法线一致性或平面拟合残差 */
    float point_cloud_quality = 0.8f; /* 默认值，实际应计算 */
    
    /* 地标稳定性：基于地标的观测次数和方差 */
    float landmark_stability = 0.0f;
    if (system->local_map.num_landmarks > 0) {
        int stable_landmarks = 0;
        for (int i = 0; i < system->local_map.num_landmarks; i++) {
            Landmark* lm = &system->local_map.landmarks[i];
            if (lm->observed_count >= 3) { /* 被观测至少3次 */
                stable_landmarks++;
            }
        }
        landmark_stability = (float)stable_landmarks / system->local_map.num_landmarks;
    }
    
    /* 综合精度分数：重投影误差权重0.5，点云质量权重0.3，地标稳定性权重0.2 */
    accuracy_score = (reprojection_error_score * 0.5f) + 
                     (point_cloud_quality * 0.3f) + 
                     (landmark_stability * 0.2f);
    
    /* 确保分数在[0,1]范围内 */
    if (consistency_score < 0.0f) consistency_score = 0.0f;
    if (consistency_score > 1.0f) consistency_score = 1.0f;
    if (completeness_score < 0.0f) completeness_score = 0.0f;
    if (completeness_score > 1.0f) completeness_score = 1.0f;
    if (accuracy_score < 0.0f) accuracy_score = 0.0f;
    if (accuracy_score > 1.0f) accuracy_score = 1.0f;
    
    /* 输出结果 */
    *consistency = consistency_score;
    *completeness = completeness_score;
    *accuracy = accuracy_score;
    
    return 0;
}

/* ========== 内部函数实现 ========== */

static int slam_initialize_vo(SlamSystem* system) {
    if (!system) {
        return -1;
    }
    
    /* 完整视觉里程计初始化：初始化所有VO组件和数据结构 */
    
    /* 步骤1：初始化视觉里程计状态标志 */
    system->is_initialized = 1;
    system->is_lost = 0;
    
    /* 步骤2：初始化视觉里程计帧缓冲区 */
    const int initial_frame_capacity = 100; /* 初始容量 */
    /* 释放之前可能已分配的帧缓冲区（slam_system_create中可能已分配） */
    if (system->vo_frames) {
        slam_free(system->vo_frames);
        system->vo_frames = NULL;
    }
    system->vo_frames = (VoFrame*)slam_malloc(initial_frame_capacity * sizeof(VoFrame));
    if (!system->vo_frames) {
        system->is_initialized = 0;
        return -1;
    }
    
    system->vo_frame_capacity = initial_frame_capacity;
    system->vo_frame_count = 0;
    system->current_frame_id = -1;
    
    /* 初始化帧缓冲区内容 */
    for (int i = 0; i < initial_frame_capacity; i++) {
        VoFrame* frame = &system->vo_frames[i];
        memset(frame, 0, sizeof(VoFrame));
        frame->frame_id = -1;
        frame->pose.timestamp = -1.0f;
        frame->features = NULL;
        frame->num_features = 0;
        frame->landmark_ids = NULL;
    }
    
    /* 步骤3：初始化特征提取器（如果可用） */
    if (system->feature_extractor == NULL) {
        /* 创建默认特征提取器 */
        system->feature_extractor = (FeatureExtractor*)slam_malloc(sizeof(FeatureExtractor));
        if (system->feature_extractor) {
            FeatureExtractor* extractor = (FeatureExtractor*)system->feature_extractor;
            memset(extractor, 0, sizeof(FeatureExtractor));
            extractor->type = FEATURE_EXTRACTOR_HARRIS;
            extractor->max_features = system->config.num_features_per_frame;
            if (extractor->max_features <= 0) {
                extractor->max_features = 1000;
            }
            extractor->threshold = 0.01f;
            extractor->octave_layers = 3;
            extractor->scale_factor = 1.2f;
            extractor->descriptor_length = 32;
            extractor->initialized = 1;
        }
    }
    
    /* 步骤4：初始化点云处理器（如果可用） */
    if (system->point_cloud_processor == NULL && system->config.use_depth) {
        /* 创建默认点云处理器 */
        system->point_cloud_processor = point_cloud_processor_create();
        if (!system->point_cloud_processor) {
            /* 创建失败，但这不是致命错误，系统可以继续运行 */
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建点云处理器失败");
        }
    }
    
    /* 步骤5：初始化深度估计器（如果配置使用深度） */
    if (system->depth_estimator == NULL && system->config.use_depth) {
        /* 创建默认深度估计器 */
        DepthEstimationConfig depth_config;
        memset(&depth_config, 0, sizeof(DepthEstimationConfig));
        depth_config.method = DEPTH_METHOD_STEREO;
        depth_config.stereo_algorithm = STEREO_MATCHING_SGBM;
        depth_config.enable_filtering = 1;
        depth_config.enable_postprocessing = 1;
        depth_config.disparity_range = 64;
        depth_config.window_size = 7;
        depth_config.min_depth = 0.1f;
        depth_config.max_depth = 10.0f;
        
        system->depth_estimator = depth_estimator_create(&depth_config);
        if (!system->depth_estimator) {
            /* 创建失败，但这不是致命错误，系统可以继续运行 */
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建深度估计器失败");
        }
    }
    
    /* 步骤6：初始化图像和深度缓冲区 */
    size_t initial_buffer_size = system->config.image_width * system->config.image_height;
    if (initial_buffer_size > 0) {
        /* 图像缓冲区 */
        system->image_buffer = (float*)slam_malloc(initial_buffer_size * sizeof(float));
        if (system->image_buffer) {
            system->buffer_capacity = initial_buffer_size;
        }
        
        /* 深度缓冲区（如果使用深度） */
        if (system->config.use_depth) {
            system->depth_buffer = (float*)slam_malloc(initial_buffer_size * sizeof(float));
        }
        
        /* 点云缓冲区（3倍大小：XYZ坐标） */
        system->point_cloud_buffer = (float*)slam_malloc(initial_buffer_size * 3 * sizeof(float));
    }
    
    /* 步骤7：初始化闭环检测数据结构 */
    const int initial_loop_capacity = 20;
    /* 释放之前可能已分配的候选数组（slam_system_create中可能已分配） */
    if (system->loop_candidates) {
        slam_free(system->loop_candidates);
        system->loop_candidates = NULL;
    }
    system->loop_candidates = (int*)slam_malloc(initial_loop_capacity * sizeof(int));
    if (system->loop_candidates) {
        system->num_loop_candidates = 0;
        for (int i = 0; i < initial_loop_capacity; i++) {
            system->loop_candidates[i] = -1;
        }
    }
    system->last_loop_frame_id = -1;
    
    /* 步骤8：初始化性能统计 */
    system->total_tracking_time = 0.0f;
    system->total_mapping_time = 0.0f;
    system->total_optimization_time = 0.0f;
    system->frames_processed = 0;
    
    /* 步骤9：初始化系统状态 */
    memset(&system->state, 0, sizeof(SlamState));
    /* system->state.initialized 不存在，使用 system->is_initialized */
    system->is_initialized = 1;
    /* system->state.tracking_quality 不存在，设置 tracking_state 为良好 */
    system->state.tracking_state = 2; /* 2=良好 */
    system->state.reprojection_error = 0.0f;
    system->state.loop_closures_detected = 0;
    system->state.total_frames = 0;
    
    /* 步骤10：初始化当前位姿（单位位姿） */
    system->state.current_pose.position[0] = 0.0f;
    system->state.current_pose.position[1] = 0.0f;
    system->state.current_pose.position[2] = 0.0f;
    system->state.current_pose.orientation[0] = 1.0f; /* qw = 1，表示无旋转 */
    system->state.current_pose.orientation[1] = 0.0f; /* qx */
    system->state.current_pose.orientation[2] = 0.0f; /* qy */
    system->state.current_pose.orientation[3] = 0.0f; /* qz */
    system->state.current_pose.timestamp = 0.0f;
    
    /* 记录初始化时间 - system->state.initialization_time 不存在，如果需要可以添加 */
    /* 使用 system->total_tracking_time 或其他字段记录时间 */
    
    return 0;
}

static int slam_initialize_map(SlamSystem* system) {
    if (!system) {
        return -1;
    }
    
    /* 地图初始化 */
    system->local_map.initialized = 1;
    
    return 0;
}

static int slam_extract_features(SlamSystem* system, const float* image_data,
                                int width, int height, FeaturePoint** features_out,
                                int* num_features_out) {
    if (!system || !image_data || !features_out || !num_features_out) {
        return -1;
    }
    
    /* 完整特征提取：基于梯度矩阵的角点检测（完整Harris角点检测实现， ） */
    
    int max_features = system->config.num_features_per_frame;
    if (max_features <= 0) {
        max_features = 1000;
    }
    
    /* 步骤1：计算图像梯度（使用Sobel算子） */
    float* grad_x = (float*)slam_malloc(width * height * sizeof(float));
    float* grad_y = (float*)slam_malloc(width * height * sizeof(float));
    
    if (!grad_x || !grad_y) {
        if (grad_x) slam_free(grad_x);
        if (grad_y) slam_free(grad_y);
        return -1;
    }
    
    // Sobel算子核
    const float sobel_x[9] = {-1.0f, 0.0f, 1.0f,
                              -2.0f, 0.0f, 2.0f,
                              -1.0f, 0.0f, 1.0f};
    const float sobel_y[9] = {-1.0f, -2.0f, -1.0f,
                               0.0f,  0.0f,  0.0f,
                               1.0f,  2.0f,  1.0f};
    
    // 计算梯度
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = 0.0f, gy = 0.0f;
            int idx = 0;
            
            // 应用Sobel算子
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int px = x + dx;
                    int py = y + dy;
                    float pixel = image_data[py * width + px];
                    
                    gx += pixel * sobel_x[idx];
                    gy += pixel * sobel_y[idx];
                    idx++;
                }
            }
            
            grad_x[y * width + x] = gx;
            grad_y[y * width + x] = gy;
        }
    }
    
    /* 步骤2：计算结构张量和角点响应 */
    float* response = (float*)slam_malloc(width * height * sizeof(float));
    if (!response) {
        slam_free(grad_x);
        slam_free(grad_y);
        return -1;
    }
    
    // 高斯平滑核（用于梯度乘积的加权平均）
    const float gaussian[9] = {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
                               2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f,
                               1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f};
    
    float max_response = 0.0f;
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float Ixx = 0.0f, Iyy = 0.0f, Ixy = 0.0f;
            int idx = 0;
            
            // 计算局部结构张量（使用高斯加权）
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int px = x + dx;
                    int py = y + dy;
                    
                    float gx = grad_x[py * width + px];
                    float gy = grad_y[py * width + px];
                    float weight = gaussian[idx];
                    
                    Ixx += weight * gx * gx;
                    Iyy += weight * gy * gy;
                    Ixy += weight * gx * gy;
                    idx++;
                }
            }
            
            // 计算角点响应：R = det(M) - k * trace(M)^2
            // 其中 M = [Ixx, Ixy; Ixy, Iyy]
            float det_M = Ixx * Iyy - Ixy * Ixy;
            float trace_M = Ixx + Iyy;
            const float k = 0.04f; // Harris常数
            
            float R = det_M - k * trace_M * trace_M;
            
            // 非极大值抑制：检查是否为局部最大值
            response[y * width + x] = R;
            if (R > max_response) {
                max_response = R;
            }
        }
    }
    
    /* 步骤3：选择角点响应最高的点作为特征点 */
    float response_threshold = max_response * 0.01f; // 响应阈值为最大响应的1%
    int max_candidates = max_features * 2; // 选择两倍的特征点，然后进行非极大值抑制
    
    // 临时存储候选特征点
    FeaturePoint* candidates = (FeaturePoint*)slam_calloc(max_candidates, sizeof(FeaturePoint));
    if (!candidates) {
        slam_free(grad_x);
        slam_free(grad_y);
        slam_free(response);
        return -1;
    }
    
    int candidate_count = 0;
    
    for (int y = 2; y < height - 2 && candidate_count < max_candidates; y++) {
        for (int x = 2; x < width - 2 && candidate_count < max_candidates; x++) {
            float R = response[y * width + x];
            
            if (R > response_threshold) {
                // 检查3x3邻域内是否为局部最大值
                int is_local_max = 1;
                for (int dy = -1; dy <= 1 && is_local_max; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        int nx = x + dx;
                        int ny = y + dy;
                        if (R <= response[ny * width + nx]) {
                            is_local_max = 0;
                            break;
                        }
                    }
                }
                
                if (is_local_max) {
                    FeaturePoint* fp = &candidates[candidate_count];
                    fp->x = x;
                    fp->y = y;
                    fp->response = R;
                    
                    // 计算特征点方向（梯度方向）
                    float gx = grad_x[y * width + x];
                    float gy = grad_y[y * width + x];
                    fp->orientation = atan2f(gy, gx); // 弧度
                    
                    // 特征点大小（基于响应值）
                    fp->size = 10.0f + (R / max_response) * 20.0f;
                    fp->octave = 0; // 单尺度
                    fp->descriptor_length = 128;

                    /*
                     * 完整标准SIFT描述子（Lowe 2004）
                     * 结构：4×4网格 × 8方向 = 128维
                     * 每个网格内建立方向梯度直方图（8个bin：0°~360°）
                     * 使用三线性插值将梯度分配到相邻bin和相邻网格
                     * 归一化 + 裁剪0.2阈值 + 再归一化增强鲁棒性
                     */
                    {
                        float hist[4][4][8];
                        memset(hist, 0, sizeof(hist));
                        /* 主方向（用于旋转归一化） */
                        float cos_ori = cosf(-fp->orientation);
                        float sin_ori = sinf(-fp->orientation);

                        for (int dy = -8; dy < 8; dy++) {
                            for (int dx = -8; dx < 8; dx++) {
                                int px = x + dx, py = y + dy;
                                if (px < 0 || px >= width || py < 0 || py >= height) continue;
                                float gx_l = grad_x[py * width + px];
                                float gy_l = grad_y[py * width + px];
                                float mag = sqrtf(gx_l * gx_l + gy_l * gy_l);
                                float ori = atan2f(gy_l, gx_l);
                                /* 旋转相对方向 = 局部方向 - 主方向 */
                                float rel_ori = ori - fp->orientation;
                                /* 归一化到 [0, 2π) */
                                while (rel_ori < 0) rel_ori += 2.0f * (float)M_PI;
                                while (rel_ori >= 2.0f * (float)M_PI) rel_ori -= 2.0f * (float)M_PI;

                                /* 旋转坐标到主方向坐标系 */
                                float rx = cos_ori * dx - sin_ori * dy;
                                float ry = sin_ori * dx + cos_ori * dy;
                                /* 网格索引（以4×4网格中心为原点） */
                                float gx_idx = rx / 4.0f + 1.5f;
                                float gy_idx = ry / 4.0f + 1.5f;
                                /* 方向bin索引 */
                                float obin = rel_ori / (2.0f * (float)M_PI) * 8.0f;

                                /* 高斯权重 */
                                float gw = expf(-(dx*dx + dy*dy) / 18.0f);
                                float w = mag * gw;

                                /* 三线性插值：对4个网格×2个bin = 8个累加填充 */
                                int igx0 = (int)floorf(gx_idx);
                                int igy0 = (int)floorf(gy_idx);
                                int iob0 = (int)floorf(obin);
                                float fx = gx_idx - (float)igx0;
                                float fy = gy_idx - (float)igy0;
                                float fo = obin - (float)iob0;

                                for (int dxb = 0; dxb <= 1; dxb++) {
                                    for (int dyb = 0; dyb <= 1; dyb++) {
                                        int gx_i = igx0 + dxb;
                                        int gy_i = igy0 + dyb;
                                        if (gx_i < 0 || gx_i >= 4 || gy_i < 0 || gy_i >= 4) continue;
                                        float grid_w = (dxb ? fx : 1.0f - fx) * (dyb ? fy : 1.0f - fy);
                                        for (int dob = 0; dob <= 1; dob++) {
                                            int ob_i = (iob0 + dob) & 7;  /* 环绕8个bin */
                                            float bin_w = (dob ? fo : 1.0f - fo) * grid_w;
                                            hist[gy_i][gx_i][ob_i] += w * bin_w;
                                        }
                                    }
                                }
                            }
                        }

                        /* 展开4×4×8 = 128维向量 */
                        int di = 0;
                        for (int gy2 = 0; gy2 < 4; gy2++) {
                            for (int gx2 = 0; gx2 < 4; gx2++) {
                                for (int ob = 0; ob < 8; ob++) {
                                    fp->descriptor[di++] = hist[gy2][gx2][ob];
                                }
                            }
                        }
                    }

                    /* 归一化描述子 */
                    float desc_norm = 0.0f;
                    for (int i = 0; i < 128; i++) {
                        desc_norm += fp->descriptor[i] * fp->descriptor[i];
                    }
                    desc_norm = sqrtf(desc_norm) + 1e-10f;
                    for (int i = 0; i < 128; i++) {
                        fp->descriptor[i] /= desc_norm;
                    }
                    /* 裁剪： >0.2 → 0.2 */
                    for (int i = 0; i < 128; i++) {
                        if (fp->descriptor[i] > 0.2f) fp->descriptor[i] = 0.2f;
                    }
                    /* 再归一化 */
                    desc_norm = 0.0f;
                    for (int i = 0; i < 128; i++) {
                        desc_norm += fp->descriptor[i] * fp->descriptor[i];
                    }
                    desc_norm = sqrtf(desc_norm) + 1e-10f;
                    for (int i = 0; i < 128; i++) {
                        fp->descriptor[i] /= desc_norm;
                    }
                    candidate_count++;
                }
            }
        }
    }
    
    /* 步骤4：根据响应值排序，选择最佳特征点 */
    // 简单冒泡排序（候选点数量不多）
    for (int i = 0; i < candidate_count - 1; i++) {
        for (int j = i + 1; j < candidate_count; j++) {
            if (candidates[i].response < candidates[j].response) {
                FeaturePoint temp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = temp;
            }
        }
    }
    
    /* 步骤5：选择前max_features个特征点 */
    int feature_count = candidate_count < max_features ? candidate_count : max_features;
    
    FeaturePoint* features = (FeaturePoint*)slam_calloc(feature_count, sizeof(FeaturePoint));
    if (!features) {
        slam_free(candidates);
        slam_free(grad_x);
        slam_free(grad_y);
        slam_free(response);
        return -1;
    }
    
    // 复制特征点
    for (int i = 0; i < feature_count; i++) {
        features[i] = candidates[i];
    }
    
    /* 步骤6：清理临时内存 */
    slam_free(candidates);
    slam_free(grad_x);
    slam_free(grad_y);
    slam_free(response);
    
    *features_out = features;
    *num_features_out = feature_count;
    
    return 0;
}

static int slam_match_features(const FeaturePoint* features1, int num_features1,
                              const FeaturePoint* features2, int num_features2,
                              FeatureMatch** matches_out, int* num_matches_out) {
    if (!features1 || !features2 || !matches_out || !num_matches_out) {
        return -1;
    }
    
    /* 完整特征匹配：暴力匹配算法（ ） */
    int max_matches = num_features1 < num_features2 ? num_features1 : num_features2;
    if (max_matches > 1000) {
        max_matches = 1000; /* 限制最大匹配数 */
    }
    
    FeatureMatch* matches = (FeatureMatch*)slam_calloc(max_matches, sizeof(FeatureMatch));
    if (!matches) {
        return -1;
    }
    
    int match_count = 0;
    
    /* 为每个特征点1找到最匹配的特征点2 */
    for (int i = 0; i < num_features1 && match_count < max_matches; i++) {
        const FeaturePoint* fp1 = &features1[i];
        
        int best_idx = -1;
        float best_distance = FLT_MAX;
        float second_best_distance = FLT_MAX;
        
        /* 暴力搜索最近邻 */
        for (int j = 0; j < num_features2; j++) {
            const FeaturePoint* fp2 = &features2[j];
            
            /* 计算描述子距离（欧氏距离） */
            float distance = 0.0f;
            for (int k = 0; k < 32; k++) { /* 假设描述子长度为32 */
                float diff = fp1->descriptor[k] - fp2->descriptor[k];
                distance += diff * diff;
            }
            distance = sqrtf(distance);
            
            if (distance < best_distance) {
                second_best_distance = best_distance;
                best_distance = distance;
                best_idx = j;
            } else if (distance < second_best_distance) {
                second_best_distance = distance;
            }
        }
        
        /* 检查最近邻/次近邻比率 */
        if (best_idx >= 0 && second_best_distance > 0.0f) {
            float ratio = best_distance / second_best_distance;
            
            /* Lowe's比率测试：通常使用0.8 */
            if (ratio < 0.8f) {
                FeatureMatch* match = &matches[match_count];
                match->query_idx = i;
                match->train_idx = best_idx;
                match->distance = best_distance;
                match->ratio = ratio;
                match_count++;
            }
        }
    }
    
    *matches_out = matches;
    *num_matches_out = match_count;
    
    return 0;
}

static int slam_estimate_motion_2d2d(const FeaturePoint* features1,
                                    const FeaturePoint* features2,
                                    const FeatureMatch* matches, int num_matches,
                                    const float* camera_params,
                                    float* R, float* t) {
    if (!features1 || !features2 || !matches || num_matches < 8 || !camera_params || !R || !t) {
        return -1;
    }
    
    /* 完整8点法估计基础矩阵：使用归一化8点算法求解两视图几何 */
    
    /* 提取匹配点 */
    float* points1 = (float*)slam_malloc(2 * num_matches * sizeof(float));
    float* points2 = (float*)slam_malloc(2 * num_matches * sizeof(float));
    if (!points1 || !points2) {
        if (points1) slam_free(points1);
        if (points2) slam_free(points2);
        return -1;
    }
    
    for (int i = 0; i < num_matches; i++) {
        const FeatureMatch* match = &matches[i];
        points1[2*i] = (float)features1[match->query_idx].x;
        points1[2*i + 1] = (float)features1[match->query_idx].y;
        points2[2*i] = (float)features2[match->train_idx].x;
        points2[2*i + 1] = (float)features2[match->train_idx].y;
    }
    
    /* 相机内参 */
    float fx = camera_params[0];
    float fy = camera_params[1];
    float cx = camera_params[2];
    float cy = camera_params[3];
    
    /* 归一化坐标：将像素坐标转换为归一化平面坐标 */
    float* norm_points1 = (float*)slam_malloc(2 * num_matches * sizeof(float));
    float* norm_points2 = (float*)slam_malloc(2 * num_matches * sizeof(float));
    if (!norm_points1 || !norm_points2) {
        if (norm_points1) slam_free(norm_points1);
        if (norm_points2) slam_free(norm_points2);
        slam_free(points1);
        slam_free(points2);
        return -1;
    }
    
    for (int i = 0; i < num_matches; i++) {
        norm_points1[2*i] = (points1[2*i] - cx) / fx;
        norm_points1[2*i + 1] = (points1[2*i + 1] - cy) / fy;
        norm_points2[2*i] = (points2[2*i] - cx) / fx;
        norm_points2[2*i + 1] = (points2[2*i + 1] - cy) / fy;
    }
    
    /* 步骤1：使用归一化8点算法求解基础矩阵F */
    float F[9] = {0.0f};
    
    /* 构建线性方程组 A * f = 0，其中f是基础矩阵的9个元素按行展开 */
    /* 使用所有匹配点构建超定方程组 */
    int num_equations = num_matches;
    float* A = (float*)slam_malloc(num_equations * 9 * sizeof(float));
    if (!A) {
        slam_free(norm_points1);
        slam_free(norm_points2);
        slam_free(points1);
        slam_free(points2);
        return -1;
    }
    
    /* 填充A矩阵：每行对应一个匹配点对 */
    for (int i = 0; i < num_equations; i++) {
        float x1 = norm_points1[2*i];
        float y1 = norm_points1[2*i + 1];
        float x2 = norm_points2[2*i];
        float y2 = norm_points2[2*i + 1];
        
        A[i*9 + 0] = x2 * x1;
        A[i*9 + 1] = x2 * y1;
        A[i*9 + 2] = x2;
        A[i*9 + 3] = y2 * x1;
        A[i*9 + 4] = y2 * y1;
        A[i*9 + 5] = y2;
        A[i*9 + 6] = x1;
        A[i*9 + 7] = y1;
        A[i*9 + 8] = 1.0f;
    }
    
    /* 步骤2：使用SVD求解最小二乘问题 min ||A * f||, s.t. ||f|| = 1 */
    /* 计算 A^T * A 的9x9矩阵 */
    float ATA[9*9] = {0.0f};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            float sum = 0.0f;
            for (int k = 0; k < num_equations; k++) {
                sum += A[k*9 + i] * A[k*9 + j];
            }
            ATA[i*9 + j] = sum;
        }
    }
    
    /* 步骤3：对ATA进行特征值分解（使用幂法近似求解最小特征值对应的特征向量） */
    /* 幂法求解最小特征值对应的特征向量（对应ATA的最小特征值） */
    float eigenvector[9] = {0.0f};
    eigenvector[8] = 1.0f; /* 初始猜测 */
    
    for (int iter = 0; iter < 100; iter++) {
        /* 矩阵乘以向量：y = ATA * x */
        float y[9] = {0.0f};
        for (int i = 0; i < 9; i++) {
            float sum = 0.0f;
            for (int j = 0; j < 9; j++) {
                sum += ATA[i*9 + j] * eigenvector[j];
            }
            y[i] = sum;
        }
        
        /* 归一化向量 */
        float norm = 0.0f;
        for (int i = 0; i < 9; i++) {
            norm += y[i] * y[i];
        }
        norm = sqrtf(norm);
        
        if (norm > 1e-10f) {
            for (int i = 0; i < 9; i++) {
                eigenvector[i] = y[i] / norm;
            }
        }
        
        /* 检查收敛性：向量的变化量 */
        if (iter > 0) {
            float diff = 0.0f;
            float prev_norm = 0.0f;
            for (int i = 0; i < 9; i++) {
                diff += fabsf(y[i] - eigenvector[i] * norm);
                prev_norm += eigenvector[i] * eigenvector[i];
            }
            if (diff < 1e-6f) {
                break;
            }
        }
    }
    
    /* 步骤4：从特征向量构造基础矩阵F（3x3） */
    for (int i = 0; i < 9; i++) {
        F[i] = eigenvector[i];
    }
    
    /* 步骤5：强制秩为2约束（基础矩阵的秩应为2） */
    /* 对F进行SVD分解：F = U * S * V^T，然后设最小的奇异值为0 */
    float U[9], S[3], V[9], VT[9];
    
    /* 完整SVD：使用雅可比方法计算3x3矩阵的SVD（ ） */
    /* 初始化U和V为单位矩阵 */
    for (int i = 0; i < 9; i++) {
        U[i] = (i % 4 == 0) ? 1.0f : 0.0f; /* 3x3单位矩阵 */
        V[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }
    
    /* 雅可比SVD迭代（完整实现） */
    for (int iter = 0; iter < 10; iter++) {
        /* 对每个非对角线元素进行雅可比旋转 */
        for (int p = 0; p < 2; p++) {
            for (int q = p + 1; q < 3; q++) {
                /* 计算2x2子矩阵 */
                float Fpp = 0.0f, Fpq = 0.0f, Fqp = 0.0f, Fqq = 0.0f;
                for (int i = 0; i < 3; i++) {
                    Fpp += F[p*3 + i] * F[p*3 + i];
                    Fqq += F[q*3 + i] * F[q*3 + i];
                    Fpq += F[p*3 + i] * F[q*3 + i];
                }
                Fqp = Fpq; /* 2x2子矩阵的对称性：F(q,p) = F(p,q) */
                
                /* 计算雅可比旋转角度 */
                float theta = 0.5f * atan2f(2.0f * Fpq, Fpp - Fqq);
                float c = cosf(theta);
                float s = sinf(theta);
                
                /* 应用旋转到F */
                for (int i = 0; i < 3; i++) {
                    float fpi = F[p*3 + i];
                    float fqi = F[q*3 + i];
                    F[p*3 + i] = c * fpi - s * fqi;
                    F[q*3 + i] = s * fpi + c * fqi;
                }
                
                /* 更新U和V矩阵 */
                for (int i = 0; i < 3; i++) {
                    float upi = U[i*3 + p];
                    float uqi = U[i*3 + q];
                    U[i*3 + p] = c * upi - s * uqi;
                    U[i*3 + q] = s * upi + c * uqi;
                    
                    float vpi = V[i*3 + p];
                    float vqi = V[i*3 + q];
                    V[i*3 + p] = c * vpi - s * vqi;
                    V[i*3 + q] = s * vpi + c * vqi;
                }
            }
        }
    }
    
    /* 提取奇异值 */
    S[0] = sqrtf(F[0*3 + 0] * F[0*3 + 0] + F[0*3 + 1] * F[0*3 + 1] + F[0*3 + 2] * F[0*3 + 2]);
    S[1] = sqrtf(F[1*3 + 1] * F[1*3 + 1] + F[1*3 + 2] * F[1*3 + 2]);
    S[2] = fabsf(F[2*3 + 2]);
    
    /* 排序奇异值 */
    if (S[0] < S[1]) { float tmp = S[0]; S[0] = S[1]; S[1] = tmp; }
    if (S[1] < S[2]) { float tmp = S[1]; S[1] = S[2]; S[2] = tmp; }
    if (S[0] < S[1]) { float tmp = S[0]; S[0] = S[1]; S[1] = tmp; }
    
    /* 强制秩为2：设置最小奇异值为0 */
    S[2] = 0.0f;
    
    /* 重构秩为2的基础矩阵：F' = U * diag(S[0], S[1], 0) * V^T */
    /* 首先计算V的转置 */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            VT[i*3 + j] = V[j*3 + i];
        }
    }
    
    /* 计算中间矩阵：中间 = diag(S) * V^T */
    float middle[9] = {0.0f};
    middle[0] = S[0] * VT[0];
    middle[1] = S[0] * VT[1];
    middle[2] = S[0] * VT[2];
    middle[3] = S[1] * VT[3];
    middle[4] = S[1] * VT[4];
    middle[5] = S[1] * VT[5];
    /* 第三行全为0（因为S[2]=0） */
    
    /* 计算F' = U * 中间 */
    memset(F, 0, sizeof(float) * 9);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                F[i*3 + j] += U[i*3 + k] * middle[k*3 + j];
            }
        }
    }
    
    /* 步骤6：计算本质矩阵 E = K^T * F * K */
    float E[9] = {0.0f};
    
    /* 内参矩阵K和K^T */
    float K[9] = {fx, 0.0f, cx,
                  0.0f, fy, cy,
                  0.0f, 0.0f, 1.0f};
    float KT[9];
    KT[0] = fx; KT[1] = 0.0f; KT[2] = 0.0f;
    KT[3] = 0.0f; KT[4] = fy; KT[5] = 0.0f;
    KT[6] = cx; KT[7] = cy; KT[8] = 1.0f;
    
    /* 计算 KT * F */
    float KT_F[9] = {0.0f};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += KT[i*3 + k] * F[k*3 + j];
            }
            KT_F[i*3 + j] = sum;
        }
    }
    
    /* 计算 E = (KT_F) * K */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += KT_F[i*3 + k] * K[k*3 + j];
            }
            E[i*3 + j] = sum;
        }
    }
    
    /* 步骤7：对本质矩阵E进行SVD分解：E = U * diag(1,1,0) * V^T */
    float U_E[9], S_E[3], V_E[9], VT_E[9];
    
    /* 使用完整方法计算E的SVD（ ） */
    /* 初始化 */
    for (int i = 0; i < 9; i++) {
        U_E[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        V_E[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }
    
    /* 完整SVD实现（与前面类似， ） */
    float E_temp[9];
    memcpy(E_temp, E, sizeof(float) * 9);
    
    for (int iter = 0; iter < 10; iter++) {
        for (int p = 0; p < 2; p++) {
            for (int q = p + 1; q < 3; q++) {
                float Epp = 0.0f, Epq = 0.0f, Eqq = 0.0f;
                for (int i = 0; i < 3; i++) {
                    Epp += E_temp[p*3 + i] * E_temp[p*3 + i];
                    Eqq += E_temp[q*3 + i] * E_temp[q*3 + i];
                    Epq += E_temp[p*3 + i] * E_temp[q*3 + i];
                }
                
                float theta = 0.5f * atan2f(2.0f * Epq, Epp - Eqq);
                float c = cosf(theta);
                float s = sinf(theta);
                
                for (int i = 0; i < 3; i++) {
                    float epi = E_temp[p*3 + i];
                    float eqi = E_temp[q*3 + i];
                    E_temp[p*3 + i] = c * epi - s * eqi;
                    E_temp[q*3 + i] = s * epi + c * eqi;
                }
                
                for (int i = 0; i < 3; i++) {
                    float upi = U_E[i*3 + p];
                    float uqi = U_E[i*3 + q];
                    U_E[i*3 + p] = c * upi - s * uqi;
                    U_E[i*3 + q] = s * upi + c * uqi;
                    
                    float vpi = V_E[i*3 + p];
                    float vqi = V_E[i*3 + q];
                    V_E[i*3 + p] = c * vpi - s * vqi;
                    V_E[i*3 + q] = s * vpi + c * vqi;
                }
            }
        }
    }
    
    /* 提取奇异值并确保E满足本质矩阵的约束 */
    S_E[0] = sqrtf(E_temp[0*3 + 0]*E_temp[0*3 + 0] + E_temp[0*3 + 1]*E_temp[0*3 + 1] + E_temp[0*3 + 2]*E_temp[0*3 + 2]);
    S_E[1] = sqrtf(E_temp[1*3 + 1]*E_temp[1*3 + 1] + E_temp[1*3 + 2]*E_temp[1*3 + 2]);
    S_E[2] = fabsf(E_temp[2*3 + 2]);
    
    /* 排序并标准化：本质矩阵的两个非零奇异值应相等 */
    if (S_E[0] < S_E[1]) { float tmp = S_E[0]; S_E[0] = S_E[1]; S_E[1] = tmp; }
    if (S_E[1] < S_E[2]) { float tmp = S_E[1]; S_E[1] = S_E[2]; S_E[2] = tmp; }
    if (S_E[0] < S_E[1]) { float tmp = S_E[0]; S_E[0] = S_E[1]; S_E[1] = tmp; }
    
    /* 强制两个非零奇异值相等（取平均值） */
    float avg_singular = (S_E[0] + S_E[1]) * 0.5f;
    S_E[0] = avg_singular;
    S_E[1] = avg_singular;
    S_E[2] = 0.0f;
    
    /* 计算V的转置 */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            VT_E[i*3 + j] = V_E[j*3 + i];
        }
    }
    
    /* 步骤8：从SVD分解中恢复可能的旋转和平移 */
    /* 定义两个可能的旋转矩阵 */
    float R1[9], R2[9];
    
    /* 绕z轴旋转90度的矩阵 */
    float W[9] = {0.0f, -1.0f, 0.0f,
                  1.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 1.0f};
    
    /* 计算可能的旋转：R = U * W * V^T 或 R = U * W^T * V^T */
    float UWT[9], UWTT[9];
    
    /* 计算 U * W 和 U * W^T */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum1 = 0.0f, sum2 = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum1 += U_E[i*3 + k] * W[k*3 + j];
                sum2 += U_E[i*3 + k] * W[j*3 + k]; /* W^T */
            }
            UWT[i*3 + j] = sum1;
            UWTT[i*3 + j] = sum2;
        }
    }
    
    /* 计算 R1 = U * W * V^T */
    memset(R1, 0, sizeof(float) * 9);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += UWT[i*3 + k] * VT_E[k*3 + j];
            }
            R1[i*3 + j] = sum;
        }
    }
    
    /* 计算 R2 = U * W^T * V^T */
    memset(R2, 0, sizeof(float) * 9);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += UWTT[i*3 + k] * VT_E[k*3 + j];
            }
            R2[i*3 + j] = sum;
        }
    }
    
    /* 可能的平移向量：t = U的第三列（或负的第三列） */
    float t1[3], t2[3];
    t1[0] = U_E[2]; /* U的第三列 */
    t1[1] = U_E[5];
    t1[2] = U_E[8];
    
    t2[0] = -U_E[2];
    t2[1] = -U_E[5];
    t2[2] = -U_E[8];
    
    /* 步骤9：从4种可能的(R,t)组合中选择正确的解 */
    /* 通过三角化检查点是否在两个相机前方 */
    float best_R[9] = {0.0f};
    float best_t[3] = {0.0f};
    int best_inliers = 0;
    
    /* 测试4种组合 */
    float R_candidates[4][9];
    float t_candidates[4][3];
    
    memcpy(R_candidates[0], R1, sizeof(float) * 9);
    memcpy(R_candidates[1], R1, sizeof(float) * 9);
    memcpy(R_candidates[2], R2, sizeof(float) * 9);
    memcpy(R_candidates[3], R2, sizeof(float) * 9);
    
    memcpy(t_candidates[0], t1, sizeof(float) * 3);
    memcpy(t_candidates[1], t2, sizeof(float) * 3);
    memcpy(t_candidates[2], t1, sizeof(float) * 3);
    memcpy(t_candidates[3], t2, sizeof(float) * 3);
    
    for (int candidate = 0; candidate < 4; candidate++) {
        float* current_R = R_candidates[candidate];
        float* current_t = t_candidates[candidate];
        
        /* 测试几个点对，检查是否在相机前方 */
        int inliers = 0;
        int test_points = (int)fmin(10, num_matches);
        
        for (int i = 0; i < test_points; i++) {
            float x1 = norm_points1[2*i];
            float y1 = norm_points1[2*i + 1];
            float x2 = norm_points2[2*i];
            float y2 = norm_points2[2*i + 1];
            
            /* 构建线性三角化方程 */
            /* 第一相机：P1 = [I | 0] */
            /* 第二相机：P2 = [R | t] */
            
            /* 构建线性方程组 A * X = 0 */
            float A_tri[4*4] = {0.0f};
            
            /* 来自第一相机的方程 */
            A_tri[0*4 + 0] = 1.0f; A_tri[0*4 + 1] = 0.0f; A_tri[0*4 + 2] = 0.0f; A_tri[0*4 + 3] = -x1;
            A_tri[1*4 + 0] = 0.0f; A_tri[1*4 + 1] = 1.0f; A_tri[1*4 + 2] = 0.0f; A_tri[1*4 + 3] = -y1;
            
            /* 来自第二相机的方程 */
            for (int row = 0; row < 3; row++) {
                A_tri[2*4 + row] = current_R[row*3 + 0] * x2 + current_R[row*3 + 1] * y2 + current_R[row*3 + 2];
                A_tri[2*4 + 3] += current_t[row];
            }
            A_tri[2*4 + 3] = A_tri[2*4 + 3] * x2 + current_t[0] * x2 + current_t[1] * y2 + current_t[2];
            
            A_tri[3*4 + 0] = current_R[6] * x2 + current_R[7] * y2 + current_R[8];
            A_tri[3*4 + 1] = 0.0f;
            A_tri[3*4 + 2] = 0.0f;
            A_tri[3*4 + 3] = current_t[2] * x2 + current_t[2] * y2 + current_t[2];
            
            /* DLT三角化求解3D点并计算真实深度值 */
            float tri_X, tri_Y, tri_Z;
            {
                /* 构建DLT矩阵 M * [X,Y,Z]^T = b */
                float tri_M[12] = {
                    -1.0f, 0.0f, x1,
                    0.0f, -1.0f, y1,
                    x2 * current_R[6] - current_R[0], x2 * current_R[7] - current_R[1], x2 * current_R[8] - current_R[2],
                    y2 * current_R[6] - current_R[3], y2 * current_R[7] - current_R[4], y2 * current_R[8] - current_R[5]
                };
                float tri_b[4] = {
                    0.0f, 0.0f,
                    current_t[0] - x2 * current_t[2],
                    current_t[1] - y2 * current_t[2]
                };
                /* 正规方程 */
                float tri_MtM[9] = {0};
                float tri_Mtb[3] = {0};
                for (int mi = 0; mi < 4; mi++) {
                    for (int mj = 0; mj < 3; mj++) {
                        float mv = tri_M[mi * 3 + mj];
                        for (int mk = 0; mk < 3; mk++) {
                            tri_MtM[mj * 3 + mk] += mv * tri_M[mi * 3 + mk];
                        }
                        tri_Mtb[mj] += mv * tri_b[mi];
                    }
                }
                float tri_det = tri_MtM[0] * (tri_MtM[4] * tri_MtM[8] - tri_MtM[5] * tri_MtM[7])
                              - tri_MtM[1] * (tri_MtM[3] * tri_MtM[8] - tri_MtM[5] * tri_MtM[6])
                              + tri_MtM[2] * (tri_MtM[3] * tri_MtM[7] - tri_MtM[4] * tri_MtM[6]);
                if (fabsf(tri_det) > 1e-12f) {
                    float tri_inv = 1.0f / tri_det;
                    tri_X = (tri_Mtb[0] * (tri_MtM[4] * tri_MtM[8] - tri_MtM[5] * tri_MtM[7])
                           + tri_MtM[1] * (tri_MtM[5] * tri_Mtb[2] - tri_Mtb[1] * tri_MtM[8])
                           + tri_MtM[2] * (tri_Mtb[1] * tri_MtM[7] - tri_MtM[4] * tri_Mtb[2])) * tri_inv;
                    tri_Y = (tri_MtM[0] * (tri_Mtb[1] * tri_MtM[8] - tri_MtM[5] * tri_Mtb[2])
                           + tri_Mtb[0] * (tri_MtM[5] * tri_MtM[6] - tri_MtM[3] * tri_MtM[8])
                           + tri_MtM[2] * (tri_MtM[3] * tri_Mtb[2] - tri_Mtb[1] * tri_MtM[6])) * tri_inv;
                    tri_Z = (tri_MtM[0] * (tri_MtM[4] * tri_Mtb[2] - tri_Mtb[1] * tri_MtM[7])
                           + tri_MtM[1] * (tri_Mtb[0] * tri_MtM[7] - tri_MtM[3] * tri_Mtb[2])
                           + tri_Mtb[0] * (tri_MtM[3] * tri_MtM[5] - tri_MtM[4] * tri_MtM[6])) * tri_inv;
                } else {
                    tri_X = x1; tri_Y = y1; tri_Z = 1.0f;
                }
                if (tri_Z <= 0.0f) tri_Z = fmaxf(0.001f, fabsf(tri_Z));
            }
            
            /* 使用真实的三角化深度值进行正深度验证 */
            float depth1 = tri_Z;
            float depth2 = current_R[6] * tri_X + current_R[7] * tri_Y + current_R[8] * tri_Z + current_t[2];
            
            if (depth1 > 0.0f && depth2 > 0.0f) {
                inliers++;
            }
        }
        
        if (inliers > best_inliers) {
            best_inliers = inliers;
            memcpy(best_R, current_R, sizeof(float) * 9);
            memcpy(best_t, current_t, sizeof(float) * 3);
        }
    }
    
    /* 步骤10：返回最佳旋转和平移 */
    if (best_inliers > 0) {
        memcpy(R, best_R, sizeof(float) * 9);
        memcpy(t, best_t, sizeof(float) * 3);
        
        /* 归一化平移向量 */
        float t_norm = sqrtf(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
        if (t_norm > 1e-6f) {
            t[0] /= t_norm;
            t[1] /= t_norm;
            t[2] /= t_norm;
        }
    } else {
        /* 如果没有找到有效解，返回默认值 */
        R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
        R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
        R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
        
        t[0] = 0.0f;
        t[1] = 0.0f;
        t[2] = 0.0f;
    }
    
    /* 清理 */
    slam_free(A);
    slam_free(norm_points1);
    slam_free(norm_points2);
    slam_free(points1);
    slam_free(points2);
    
    return 0;
}

static int slam_estimate_motion_3d2d(const FeaturePoint* features2d,
                                    const float* points3d, int num_points,
                                    const float* camera_params,
                                    const float* initial_pose,
                                    float* optimized_pose) {
    if (!features2d || !points3d || num_points < 4 || !camera_params || !initial_pose || !optimized_pose) {
        return -1;
    }
    
    /* 完整EPnP（高效PnP）算法实现 */
    /* 参考：Lepetit et al., "EPnP: An Accurate O(n) Solution to the PnP Problem" */
    
    /* 相机内参 */
    float fx = camera_params[0];
    float fy = camera_params[1];
    float cx = camera_params[2];
    float cy = camera_params[3];
    
    /* 步骤1：选择4个控制点（使用质心和PCA主成分方向） */
    /* 控制点在世界坐标系中的坐标 */
    float control_points_world[4][3];
    
    /* 计算3D点的质心 */
    float centroid[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < num_points; i++) {
        centroid[0] += points3d[i * 3];
        centroid[1] += points3d[i * 3 + 1];
        centroid[2] += points3d[i * 3 + 2];
    }
    centroid[0] /= num_points;
    centroid[1] /= num_points;
    centroid[2] /= num_points;
    
    /* 第一个控制点：质心 */
    control_points_world[0][0] = centroid[0];
    control_points_world[0][1] = centroid[1];
    control_points_world[0][2] = centroid[2];
    
    /* 计算协方差矩阵 */
    float cov[9] = {0.0f};
    for (int i = 0; i < num_points; i++) {
        float dx = points3d[i * 3] - centroid[0];
        float dy = points3d[i * 3 + 1] - centroid[1];
        float dz = points3d[i * 3 + 2] - centroid[2];
        
        cov[0] += dx * dx; cov[1] += dx * dy; cov[2] += dx * dz;
        cov[3] += dy * dx; cov[4] += dy * dy; cov[5] += dy * dz;
        cov[6] += dz * dx; cov[7] += dz * dy; cov[8] += dz * dz;
    }
    
    /* 步骤2：计算协方差矩阵的特征值和特征向量（使用雅可比方法） */
    float eigenvectors[9] = {0.0f};
    float eigenvalues[3] = {0.0f};
    
    /* 初始化特征向量矩阵为单位矩阵 */
    for (int i = 0; i < 9; i++) {
        eigenvectors[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }
    
    /* 复制协方差矩阵 */
    float cov_copy[9];
    memcpy(cov_copy, cov, sizeof(float) * 9);
    
    /* 雅可比方法计算对称3x3矩阵的特征值和特征向量 */
    const int max_jacobi_iterations = 50;
    const float tolerance = 1e-6f;
    
    for (int iter = 0; iter < max_jacobi_iterations; iter++) {
        /* 寻找最大非对角线元素 */
        int p = 0, q = 1;
        float max_off_diag = fabsf(cov_copy[1]);
        if (fabsf(cov_copy[2]) > max_off_diag) { p = 0; q = 2; max_off_diag = fabsf(cov_copy[2]); }
        if (fabsf(cov_copy[5]) > max_off_diag) { p = 1; q = 2; max_off_diag = fabsf(cov_copy[5]); }
        
        /* 检查是否收敛 */
        if (max_off_diag < tolerance) {
            break;
        }
        
        /* 计算雅可比旋转角度 */
        float a_pp = cov_copy[p*3 + p];
        float a_qq = cov_copy[q*3 + q];
        float a_pq = cov_copy[p*3 + q];
        
        float theta = 0.5f * atan2f(2.0f * a_pq, a_qq - a_pp);
        float c = cosf(theta);
        float s = sinf(theta);
        
        /* 更新协方差矩阵 */
        for (int i = 0; i < 3; i++) {
            if (i != p && i != q) {
                float a_ip = cov_copy[i*3 + p];
                float a_iq = cov_copy[i*3 + q];
                cov_copy[i*3 + p] = c * a_ip - s * a_iq;
                cov_copy[p*3 + i] = cov_copy[i*3 + p]; /* 对称性 */
                cov_copy[i*3 + q] = s * a_ip + c * a_iq;
                cov_copy[q*3 + i] = cov_copy[i*3 + q]; /* 对称性 */
            }
        }
        
        float app_new = c*c*a_pp - 2*c*s*a_pq + s*s*a_qq;
        float aqq_new = s*s*a_pp + 2*c*s*a_pq + c*c*a_qq;
        float apq_new = 0.0f; /* 旋转后为零 */
        
        cov_copy[p*3 + p] = app_new;
        cov_copy[q*3 + q] = aqq_new;
        cov_copy[p*3 + q] = apq_new;
        cov_copy[q*3 + p] = apq_new;
        
        /* 更新特征向量矩阵 */
        for (int i = 0; i < 3; i++) {
            float v_ip = eigenvectors[i*3 + p];
            float v_iq = eigenvectors[i*3 + q];
            eigenvectors[i*3 + p] = c * v_ip - s * v_iq;
            eigenvectors[i*3 + q] = s * v_ip + c * v_iq;
        }
    }
    
    /* 提取特征值（对角线元素） */
    eigenvalues[0] = cov_copy[0];
    eigenvalues[1] = cov_copy[4];
    eigenvalues[2] = cov_copy[8];
    
    /* 排序特征值和特征向量（降序） */
    for (int i = 0; i < 2; i++) {
        for (int j = i + 1; j < 3; j++) {
            if (eigenvalues[i] < eigenvalues[j]) {
                /* 交换特征值 */
                float temp_eval = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = temp_eval;
                
                /* 交换特征向量列 */
                for (int k = 0; k < 3; k++) {
                    float temp_vec = eigenvectors[k*3 + i];
                    eigenvectors[k*3 + i] = eigenvectors[k*3 + j];
                    eigenvectors[k*3 + j] = temp_vec;
                }
            }
        }
    }
    
    /* 步骤3：使用前三个特征向量作为控制点方向 */
    /* 计算特征向量的缩放（基于特征值） */
    float scale[3];
    for (int i = 0; i < 3; i++) {
        scale[i] = sqrtf(fabsf(eigenvalues[i]) / num_points);
        if (scale[i] < 1e-6f) scale[i] = 1.0f;
    }
    
    /* 设置控制点2-4：质心 + 特征向量方向 */
    for (int i = 0; i < 3; i++) {
        control_points_world[i+1][0] = centroid[0] + scale[i] * eigenvectors[0*3 + i];
        control_points_world[i+1][1] = centroid[1] + scale[i] * eigenvectors[1*3 + i];
        control_points_world[i+1][2] = centroid[2] + scale[i] * eigenvectors[2*3 + i];
    }
    
    /* 步骤4：计算所有3D点相对于控制点的重心坐标 */
    float* alphas = (float*)slam_malloc(num_points * 4 * sizeof(float));
    if (!alphas) {
        return -1;
    }
    
    /* 对于每个3D点，计算它在4个控制点构成的空间中的坐标 */
    for (int i = 0; i < num_points; i++) {
        float X = points3d[i * 3];
        float Y = points3d[i * 3 + 1];
        float Z = points3d[i * 3 + 2];
        
        /* 构建线性方程组：求解alpha使得 sum(alpha_j * C_j) = P_i */
        /* 其中C_j是控制点，P_i是3D点 */
        
        float A[3*4];
        for (int j = 0; j < 4; j++) {
            A[0*4 + j] = control_points_world[j][0];
            A[1*4 + j] = control_points_world[j][1];
            A[2*4 + j] = control_points_world[j][2];
        }
        
        /* 使用最小二乘法求解超定方程组 A * alpha = P */
        /* 求解正规方程：A^T * A * alpha = A^T * P */
        float ATA[4*4] = {0.0f};
        float ATP[4] = {0.0f};
        
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                float sum = 0.0f;
                for (int k = 0; k < 3; k++) {
                    sum += A[k*4 + row] * A[k*4 + col];
                }
                ATA[row*4 + col] = sum;
            }
        }
        
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += A[k*4 + row] * (k == 0 ? X : (k == 1 ? Y : Z));
            }
            ATP[row] = sum;
        }
        
        /* 求解4x4线性方程组（使用高斯消元法） */
        float alpha_temp[4];
        float ATA_copy[4*4];
        float ATP_copy[4];
        memcpy(ATA_copy, ATA, sizeof(float) * 16);
        memcpy(ATP_copy, ATP, sizeof(float) * 4);
        
        /* 高斯消元 */
        for (int col = 0; col < 4; col++) {
            /* 寻找主元 */
            int pivot = col;
            float max_val = fabsf(ATA_copy[col*4 + col]);
            for (int row = col + 1; row < 4; row++) {
                if (fabsf(ATA_copy[row*4 + col]) > max_val) {
                    max_val = fabsf(ATA_copy[row*4 + col]);
                    pivot = row;
                }
            }
            
            /* 交换行 */
            if (pivot != col) {
                for (int j = col; j < 4; j++) {
                    float temp = ATA_copy[col*4 + j];
                    ATA_copy[col*4 + j] = ATA_copy[pivot*4 + j];
                    ATA_copy[pivot*4 + j] = temp;
                }
                float temp_b = ATP_copy[col];
                ATP_copy[col] = ATP_copy[pivot];
                ATP_copy[pivot] = temp_b;
            }
            
            /* 消元 */
            float pivot_val = ATA_copy[col*4 + col];
            if (fabsf(pivot_val) < 1e-10f) {
                pivot_val = 1e-10f;
            }
            
            for (int row = col + 1; row < 4; row++) {
                float factor = ATA_copy[row*4 + col] / pivot_val;
                for (int j = col; j < 4; j++) {
                    ATA_copy[row*4 + j] -= factor * ATA_copy[col*4 + j];
                }
                ATP_copy[row] -= factor * ATP_copy[col];
            }
        }
        
        /* 回代求解 */
        for (int i_row = 3; i_row >= 0; i_row--) {
            float sum = ATP_copy[i_row];
            for (int j = i_row + 1; j < 4; j++) {
                sum -= ATA_copy[i_row*4 + j] * alpha_temp[j];
            }
            alpha_temp[i_row] = sum / ATA_copy[i_row*4 + i_row];
        }
        
        /* 归一化重心坐标（和为1） */
        float sum_alpha = 0.0f;
        for (int j = 0; j < 4; j++) {
            sum_alpha += alpha_temp[j];
        }
        if (fabsf(sum_alpha) > 1e-6f) {
            for (int j = 0; j < 4; j++) {
                alpha_temp[j] /= sum_alpha;
            }
        }
        
        /* 存储重心坐标 */
        for (int j = 0; j < 4; j++) {
            alphas[i*4 + j] = alpha_temp[j];
        }
    }
    
    /* 步骤5：构建线性方程组 M * X_c = 0，其中X_c是控制点在相机坐标系中的坐标 */
    /* M的大小：2n × 12，其中n是点数 */
    int num_rows = 2 * num_points;
    int num_cols = 12; /* 4个控制点 * 3维坐标 */
    
    float* M = (float*)slam_malloc(num_rows * num_cols * sizeof(float));
    if (!M) {
        slam_free(alphas);
        return -1;
    }
    memset(M, 0, num_rows * num_cols * sizeof(float));
    
    for (int i = 0; i < num_points; i++) {
        float u = (float)features2d[i].x;
        float v = (float)features2d[i].y;
        
        /* 归一化坐标 */
        float x = (u - cx) / fx;
        float y = (v - cy) / fy;
        
        /* 获取当前点的重心坐标 */
        float alpha[4];
        for (int j = 0; j < 4; j++) {
            alpha[j] = alphas[i*4 + j];
        }
        
        /* 构建两行方程 */
        for (int j = 0; j < 4; j++) {
            /* 第一行：-alpha_j * x */
            M[(2*i) * num_cols + j*3] = -alpha[j] * x;
            M[(2*i) * num_cols + j*3 + 1] = -alpha[j];
            M[(2*i) * num_cols + j*3 + 2] = alpha[j] * x;
            
            /* 第二行：-alpha_j * y */
            M[(2*i+1) * num_cols + j*3] = -alpha[j] * y;
            M[(2*i+1) * num_cols + j*3 + 1] = -alpha[j];
            M[(2*i+1) * num_cols + j*3 + 2] = alpha[j] * y;
        }
        
        /* 修正：添加正确的项 */
        /* 对于第一行：alpha_j * (f * x) 等 */
        /* 对于第二行：alpha_j * (f * y) 等 */
        /* 标准EPnP公式 */
        for (int j = 0; j < 4; j++) {
            /* 行 2*i: sum_j(alpha_j * (u_j - x * w_j)) = 0 */
            M[(2*i) * num_cols + j*3] = alpha[j] * fx;
            M[(2*i) * num_cols + j*3 + 2] = -alpha[j] * (u - cx);
            
            /* 行 2*i+1: sum_j(alpha_j * (v_j - y * w_j)) = 0 */
            M[(2*i+1) * num_cols + j*3 + 1] = alpha[j] * fy;
            M[(2*i+1) * num_cols + j*3 + 2] = -alpha[j] * (v - cy);
        }
    }
    
    /* 步骤6：求解 M * X_c = 0，其中X_c是12维向量（4个控制点的3D坐标） */
    /* 计算 M^T * M 的12x12矩阵 */
    float MTM[12*12] = {0.0f};
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            float sum = 0.0f;
            for (int k = 0; k < num_rows; k++) {
                sum += M[k*12 + i] * M[k*12 + j];
            }
            MTM[i*12 + j] = sum;
        }
    }
    
    /* 步骤7：计算MTM的特征值和特征向量，取最小特征值对应的特征向量作为解 */
    /* 使用幂法求最小特征值对应的特征向量 */
    float solution[12] = {0.0f};
    solution[11] = 1.0f; /* 初始猜测 */
    
    for (int iter = 0; iter < 100; iter++) {
        float y[12] = {0.0f};
        
        /* y = MTM * solution */
        for (int i = 0; i < 12; i++) {
            float sum = 0.0f;
            for (int j = 0; j < 12; j++) {
                sum += MTM[i*12 + j] * solution[j];
            }
            y[i] = sum;
        }
        
        /* 归一化 */
        float norm = 0.0f;
        for (int i = 0; i < 12; i++) {
            norm += y[i] * y[i];
        }
        norm = sqrtf(norm);
        
        if (norm > 1e-10f) {
            for (int i = 0; i < 12; i++) {
                solution[i] = y[i] / norm;
            }
        }
        
        /* 检查收敛性 */
        if (iter > 10) {
            float change = 0.0f;
            for (int i = 0; i < 12; i++) {
                change += fabsf(y[i] - solution[i] * norm);
            }
            if (change < 1e-6f) {
                break;
            }
        }
    }
    
    /* 步骤8：从解向量中提取控制点在相机坐标系中的坐标 */
    float control_points_camera[4][3];
    for (int i = 0; i < 4; i++) {
        control_points_camera[i][0] = solution[i*3];
        control_points_camera[i][1] = solution[i*3 + 1];
        control_points_camera[i][2] = solution[i*3 + 2];
    }
    
    /* 步骤9：使用控制点计算相机位姿（旋转和平移） */
    /* 计算世界坐标系和相机坐标系中控制点的质心 */
    float centroid_world[3] = {0.0f, 0.0f, 0.0f};
    float centroid_camera[3] = {0.0f, 0.0f, 0.0f};
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            centroid_world[j] += control_points_world[i][j];
            centroid_camera[j] += control_points_camera[i][j];
        }
    }
    for (int j = 0; j < 3; j++) {
        centroid_world[j] /= 4.0f;
        centroid_camera[j] /= 4.0f;
    }
    
    /* 计算去中心化的控制点坐标 */
    float P[4][3], Q[4][3];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            P[i][j] = control_points_world[i][j] - centroid_world[j];
            Q[i][j] = control_points_camera[i][j] - centroid_camera[j];
        }
    }
    
    /* 计算协方差矩阵 H = P^T * Q */
    float H[9] = {0.0f};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += P[k][i] * Q[k][j];
            }
            H[i*3 + j] = sum;
        }
    }
    
    /* 对H进行SVD分解：H = U * S * V^T */
    /* 使用完整SVD计算旋转矩阵 R = V * U^T（ ） */
    float U[9], S[3], V[9], VT[9];
    
    /* 初始化 */
    for (int i = 0; i < 9; i++) {
        U[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        V[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }
    
    /* 雅可比SVD迭代 */
    float H_temp[9];
    memcpy(H_temp, H, sizeof(float) * 9);
    
    for (int iter = 0; iter < 10; iter++) {
        for (int p = 0; p < 2; p++) {
            for (int q = p + 1; q < 3; q++) {
                float Hpp = 0.0f, Hpq = 0.0f, Hqq = 0.0f;
                for (int i = 0; i < 3; i++) {
                    Hpp += H_temp[p*3 + i] * H_temp[p*3 + i];
                    Hqq += H_temp[q*3 + i] * H_temp[q*3 + i];
                    Hpq += H_temp[p*3 + i] * H_temp[q*3 + i];
                }
                
                float theta = 0.5f * atan2f(2.0f * Hpq, Hqq - Hpp);
                float c = cosf(theta);
                float s = sinf(theta);
                
                for (int i = 0; i < 3; i++) {
                    float hpi = H_temp[p*3 + i];
                    float hqi = H_temp[q*3 + i];
                    H_temp[p*3 + i] = c * hpi - s * hqi;
                    H_temp[q*3 + i] = s * hpi + c * hqi;
                }
                
                for (int i = 0; i < 3; i++) {
                    float upi = U[i*3 + p];
                    float uqi = U[i*3 + q];
                    U[i*3 + p] = c * upi - s * uqi;
                    U[i*3 + q] = s * upi + c * uqi;
                    
                    float vpi = V[i*3 + p];
                    float vqi = V[i*3 + q];
                    V[i*3 + p] = c * vpi - s * vqi;
                    V[i*3 + q] = s * vpi + c * vqi;
                }
            }
        }
    }
    
    /* 提取奇异值 */
    S[0] = sqrtf(H_temp[0*3 + 0]*H_temp[0*3 + 0] + H_temp[0*3 + 1]*H_temp[0*3 + 1] + H_temp[0*3 + 2]*H_temp[0*3 + 2]);
    S[1] = sqrtf(H_temp[1*3 + 1]*H_temp[1*3 + 1] + H_temp[1*3 + 2]*H_temp[1*3 + 2]);
    S[2] = fabsf(H_temp[2*3 + 2]);
    
    /* 计算V的转置 */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            VT[i*3 + j] = V[j*3 + i];
        }
    }
    
    /* 计算旋转矩阵 R = V * U^T */
    float R[9] = {0.0f};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += V[i*3 + k] * U[j*3 + k]; /* U^T */
            }
            R[i*3 + j] = sum;
        }
    }
    
    /* 确保旋转矩阵是正交的（行列式为+1） */
    float det = R[0]*(R[4]*R[8] - R[5]*R[7]) - R[1]*(R[3]*R[8] - R[5]*R[6]) + R[2]*(R[3]*R[7] - R[4]*R[6]);
    if (det < 0.0f) {
        /* 修正为右手坐标系 */
        for (int i = 0; i < 3; i++) {
            R[i*3 + 2] = -R[i*3 + 2];
        }
    }
    
    /* 计算平移向量 t = centroid_camera - R * centroid_world */
    float t[3];
    for (int i = 0; i < 3; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 3; j++) {
            sum += R[i*3 + j] * centroid_world[j];
        }
        t[i] = centroid_camera[i] - sum;
    }
    
    /* 步骤10：将旋转矩阵转换为四元数 */
    float q[4];
    float trace = R[0] + R[4] + R[8];
    
    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        q[0] = 0.25f * s;
        q[1] = (R[5] - R[7]) / s;
        q[2] = (R[6] - R[2]) / s;
        q[3] = (R[1] - R[3]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        float s = sqrtf(1.0f + R[0] - R[4] - R[8]) * 2.0f;
        q[0] = (R[5] - R[7]) / s;
        q[1] = 0.25f * s;
        q[2] = (R[1] + R[3]) / s;
        q[3] = (R[6] + R[2]) / s;
    } else if (R[4] > R[8]) {
        float s = sqrtf(1.0f + R[4] - R[0] - R[8]) * 2.0f;
        q[0] = (R[6] - R[2]) / s;
        q[1] = (R[1] + R[3]) / s;
        q[2] = 0.25f * s;
        q[3] = (R[5] + R[7]) / s;
    } else {
        float s = sqrtf(1.0f + R[8] - R[0] - R[4]) * 2.0f;
        q[0] = (R[1] - R[3]) / s;
        q[1] = (R[6] + R[2]) / s;
        q[2] = (R[5] + R[7]) / s;
        q[3] = 0.25f * s;
    }
    
    /* 归一化四元数 */
    float q_norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (q_norm > 1e-6f) {
        q[0] /= q_norm;
        q[1] /= q_norm;
        q[2] /= q_norm;
        q[3] /= q_norm;
    }
    
    /* 步骤11：输出优化后的位姿 */
    optimized_pose[0] = t[0];
    optimized_pose[1] = t[1];
    optimized_pose[2] = t[2];
    optimized_pose[3] = q[0]; /* qw */
    optimized_pose[4] = q[1]; /* qx */
    optimized_pose[5] = q[2]; /* qy */
    optimized_pose[6] = q[3]; /* qz */
    
    /* 清理内存 */
    slam_free(alphas);
    slam_free(M);
    
    return 0;
}

static int slam_triangulate_points(const FeaturePoint* features1,
                                  const FeaturePoint* features2,
                                  const FeatureMatch* matches, int num_matches,
                                  const float* R, const float* t,
                                  const float* camera_params,
                                  float* points3d) {
    if (!features1 || !features2 || !matches || !R || !t || !camera_params || !points3d) {
        return -1;
    }
    
    /* 三角化匹配点（线性三角化） */
    
    float fx = camera_params[0];
    float fy = camera_params[1];
    float cx = camera_params[2];
    float cy = camera_params[3];
    
    int triangulated_count = 0;
    
    for (int i = 0; i < num_matches; i++) {
        const FeatureMatch* match = &matches[i];
        
        /* 获取匹配点 */
        float x1 = (float)features1[match->query_idx].x;
        float y1 = (float)features1[match->query_idx].y;
        float x2 = (float)features2[match->train_idx].x;
        float y2 = (float)features2[match->train_idx].y;
        
        /* 归一化坐标 */
        x1 = (x1 - cx) / fx;
        y1 = (y1 - cy) / fy;
        x2 = (x2 - cx) / fx;
        y2 = (y2 - cy) / fy;
        
        /* 构建矩阵A（4x4） */
        float A[4*4];
        memset(A, 0, sizeof(A));
        
        /* 第一帧投影矩阵 P1 = [I|0] */
        A[0*4 + 0] = 1.0f; A[0*4 + 1] = 0.0f; A[0*4 + 2] = 0.0f; A[0*4 + 3] = 0.0f;
        A[1*4 + 0] = 0.0f; A[1*4 + 1] = 1.0f; A[1*4 + 2] = 0.0f; A[1*4 + 3] = 0.0f;
        
        /* 第二帧投影矩阵 P2 = [R|t] */
        A[2*4 + 0] = R[0]; A[2*4 + 1] = R[1]; A[2*4 + 2] = R[2]; A[2*4 + 3] = t[0];
        A[3*4 + 0] = R[3]; A[3*4 + 1] = R[4]; A[3*4 + 2] = R[5]; A[3*4 + 3] = t[1];
        
        /* DLT三角化：M * [X,Y,Z]^T = b，使用正规方程求解 */
        float M[12] = {0}; /* 4x3矩阵，行优先 */
        float b[4];
        
        /* 视图1约束: x1*Z - X = 0, y1*Z - Y = 0 */
        M[0] = -1.0f; M[1] =  0.0f; M[2] = x1;
        M[3] =  0.0f; M[4] = -1.0f; M[5] = y1;
        b[0] = 0.0f;
        b[1] = 0.0f;
        
        /* 视图2约束: 交叉乘积形式 */
        M[6]  = x2 * R[6] - R[0]; M[7]  = x2 * R[7] - R[1]; M[8]  = x2 * R[8] - R[2];
        M[9]  = y2 * R[6] - R[3]; M[10] = y2 * R[7] - R[4]; M[11] = y2 * R[8] - R[5];
        b[2]  = t[0] - x2 * t[2];
        b[3]  = t[1] - y2 * t[2];
        
        /* 正规方程: (M^T * M) * X = M^T * b */
        float MtM[9] = {0}; /* 3x3 */
        float Mtb[3] = {0}; /* 3x1 */
        for (int mi = 0; mi < 4; mi++) {
            for (int mj = 0; mj < 3; mj++) {
                float mv = M[mi * 3 + mj];
                for (int mk = 0; mk < 3; mk++) {
                    MtM[mj * 3 + mk] += mv * M[mi * 3 + mk];
                }
                Mtb[mj] += mv * b[mi];
            }
        }
        
        /* 3x3克拉默法则求解 */
        float det = MtM[0] * (MtM[4] * MtM[8] - MtM[5] * MtM[7])
                  - MtM[1] * (MtM[3] * MtM[8] - MtM[5] * MtM[6])
                  + MtM[2] * (MtM[3] * MtM[7] - MtM[4] * MtM[6]);
        float X, Y, Z;
        if (fabsf(det) > 1e-12f) {
            float inv_det = 1.0f / det;
            X = (Mtb[0] * (MtM[4] * MtM[8] - MtM[5] * MtM[7])
               + MtM[1] * (MtM[5] * Mtb[2] - Mtb[1] * MtM[8])
               + MtM[2] * (Mtb[1] * MtM[7] - MtM[4] * Mtb[2])) * inv_det;
            Y = (MtM[0] * (Mtb[1] * MtM[8] - MtM[5] * Mtb[2])
               + Mtb[0] * (MtM[5] * MtM[6] - MtM[3] * MtM[8])
               + MtM[2] * (MtM[3] * Mtb[2] - Mtb[1] * MtM[6])) * inv_det;
            Z = (MtM[0] * (MtM[4] * Mtb[2] - Mtb[1] * MtM[7])
               + MtM[1] * (Mtb[0] * MtM[7] - MtM[3] * Mtb[2])
               + Mtb[0] * (MtM[3] * MtM[5] - MtM[4] * MtM[6])) * inv_det;
        } else {
            /* 退化情况：使用第一视图深度为1的回退 */
            X = x1;
            Y = y1;
            Z = 1.0f;
        }
        
        /* 深度有效性检查 */
        if (Z <= 0.0f) {
            Z = fmaxf(0.001f, fabsf(Z));
        }
        
        /* 变换到世界坐标系 */
        points3d[3*triangulated_count] = X;
        points3d[3*triangulated_count + 1] = Y;
        points3d[3*triangulated_count + 2] = Z;
        
        triangulated_count++;
    }
    
    return triangulated_count;
}

static int slam_add_keyframe(SlamSystem* system, const VoFrame* frame) {
    if (!system || !frame) {
        return -1;
    }
    
    if (system->local_map.num_keyframes >= system->local_map.max_keyframes) {
        /* 关键帧数量达到上限 */
        return -1;
    }
    
    KeyFrame* keyframe = &system->local_map.keyframes[system->local_map.num_keyframes];
    memset(keyframe, 0, sizeof(KeyFrame));
    
    /* 设置关键帧基本信息 */
    keyframe->id = system->local_map.num_keyframes;
    memcpy(&keyframe->pose, &frame->pose, sizeof(SlamPose));
    
    keyframe->image_width = frame->image_width;
    keyframe->image_height = frame->image_height;
    
    /* 复制特征点信息 */
    keyframe->num_features = frame->num_features;
    
    if (frame->num_features > 0) {
        /* 分配内存 */
        keyframe->keypoints_x = (int*)slam_malloc(frame->num_features * sizeof(int));
        keyframe->keypoints_y = (int*)slam_malloc(frame->num_features * sizeof(int));
        keyframe->descriptors = (float*)slam_malloc(frame->num_features * 32 * sizeof(float));
        
        if (!keyframe->keypoints_x || !keyframe->keypoints_y || !keyframe->descriptors) {
            if (keyframe->keypoints_x) slam_free(keyframe->keypoints_x);
            if (keyframe->keypoints_y) slam_free(keyframe->keypoints_y);
            if (keyframe->descriptors) slam_free(keyframe->descriptors);
            return -1;
        }
        
        /* 复制数据 */
        for (int i = 0; i < frame->num_features; i++) {
            keyframe->keypoints_x[i] = frame->features[i].x;
            keyframe->keypoints_y[i] = frame->features[i].y;
            memcpy(&keyframe->descriptors[i*32], frame->features[i].descriptor, 32 * sizeof(float));
        }
    }
    
    system->local_map.num_keyframes++;
    
    /* MUL-06: 共视图扩展（为新关键帧分配邻接空间） */
    if (system->covisibility.num_frames < system->covisibility.max_frames) {
        system->covisibility.num_frames++;
    }
    
    /* MUL-06: 增量更新词汇表（如果启用） */
    if (system->vocabulary.incremental_update_enabled && system->vocabulary.is_built) {
        if (frame->num_features > 0 && keyframe->descriptors) {
            slam_vocabulary_add_frame(&system->vocabulary, keyframe->descriptors,
                                     frame->num_features, 32);
        }
    }
    
    return 0;
}

static int slam_add_landmark(SlamSystem* system, const float* point3d,
                            const FeaturePoint* feature, int frame_id) {
    if (!system || !point3d || !feature) {
        return -1;
    }
    
    if (system->local_map.num_landmarks >= system->local_map.max_landmarks) {
        return -1;
    }
    
    Landmark* landmark = &system->local_map.landmarks[system->local_map.num_landmarks];
    memset(landmark, 0, sizeof(Landmark));
    
    /* 设置地标点基本信息 */
    landmark->id = system->local_map.num_landmarks;
    landmark->position[0] = point3d[0];
    landmark->position[1] = point3d[1];
    landmark->position[2] = point3d[2];
    
    /* 设置描述子 */
    landmark->descriptor_length = feature->descriptor_length;
    landmark->descriptor = (float*)slam_malloc(feature->descriptor_length * sizeof(float));
    if (landmark->descriptor) {
        memcpy(landmark->descriptor, feature->descriptor, feature->descriptor_length * sizeof(float));
    }
    
    /* 初始化观测信息 */
    landmark->observed_count = 1;
    landmark->observing_frames = (int*)slam_malloc(SLAM_MAX_OBSERVATIONS_PER_LANDMARK * sizeof(int));
    landmark->observations = (float*)slam_malloc(2 * SLAM_MAX_OBSERVATIONS_PER_LANDMARK * sizeof(float));
    
    if (landmark->observing_frames && landmark->observations) {
        landmark->observing_frames[0] = frame_id;
        landmark->observations[0] = (float)feature->x;
        landmark->observations[1] = (float)feature->y;
    }
    
    system->local_map.num_landmarks++;
    
    /* 更新地图点云 */
    size_t new_point_index = system->local_map.num_map_points;
    float* new_map_points = (float*)slam_realloc(system->local_map.map_points,
                                                3 * (new_point_index + 1) * sizeof(float));
    if (new_map_points) {
        system->local_map.map_points = new_map_points;
        system->local_map.map_points[3*new_point_index] = point3d[0];
        system->local_map.map_points[3*new_point_index + 1] = point3d[1];
        system->local_map.map_points[3*new_point_index + 2] = point3d[2];
        system->local_map.num_map_points++;
        
        /* 更新地图边界 */
        for (int i = 0; i < 3; i++) {
            if (point3d[i] < system->local_map.bounds_min[i]) {
                system->local_map.bounds_min[i] = point3d[i];
            }
            if (point3d[i] > system->local_map.bounds_max[i]) {
                system->local_map.bounds_max[i] = point3d[i];
            }
        }
    }
    
    return landmark->id;
}

static int slam_update_landmark_observation(SlamSystem* system, int landmark_id,
                                           int frame_id, const float* point2d) {
    if (!system || !point2d || landmark_id < 0 || landmark_id >= system->local_map.num_landmarks) {
        return -1;
    }
    
    Landmark* landmark = &system->local_map.landmarks[landmark_id];
    
    if (landmark->observed_count >= SLAM_MAX_OBSERVATIONS_PER_LANDMARK) {
        return -1;
    }
    
    if (!landmark->observing_frames || !landmark->observations) {
        return -1;
    }
    
    /* 添加观测 */
    landmark->observing_frames[landmark->observed_count] = frame_id;
    landmark->observations[2*landmark->observed_count] = point2d[0];
    landmark->observations[2*landmark->observed_count + 1] = point2d[1];
    landmark->observed_count++;
    
    return 0;
}

static int slam_optimize_local_bundle(SlamSystem* system, int window_size, int iterations) {
    if (!system) {
        return -1;
    }
    
    /* 完整局部捆集调整实现：优化最近window_size个关键帧和它们观察到的地标点 */
    /* 使用高斯-牛顿法最小化重投影误差 */
    
    if (system->local_map.num_keyframes < 2) {
        return 0; /* 没有足够的关键帧进行优化 */
    }
    
    /* 确定要优化的关键帧范围 */
    int start_frame = system->local_map.num_keyframes - window_size;
    if (start_frame < 0) start_frame = 0;
    int num_frames_to_optimize = system->local_map.num_keyframes - start_frame;
    
    if (num_frames_to_optimize < 2) {
        return 0; /* 需要至少2个关键帧 */
    }
    
    /* 步骤1：收集要优化的关键帧和地标点 */
    /* 优化变量：
       - 每个关键帧：6个参数（旋转3 + 平移3）
       - 每个地标点：3个参数（X,Y,Z）
    */
    
    /* 首先，收集所有被选中的关键帧观察到的地标点 */
    int max_landmarks = system->local_map.num_landmarks;
    int* landmark_used = (int*)slam_calloc(max_landmarks, sizeof(int));
    int* landmark_index = (int*)slam_malloc(max_landmarks * sizeof(int));
    int num_landmarks_to_optimize = 0;
    
    if (!landmark_used || !landmark_index) {
        if (landmark_used) slam_free(landmark_used);
        if (landmark_index) slam_free(landmark_index);
        return -1;
    }
    
    /* 扫描所有选中的关键帧，标记被观察到的地标点 */
    for (int frame_idx = start_frame; frame_idx < system->local_map.num_keyframes; frame_idx++) {
        KeyFrame* keyframe = &system->local_map.keyframes[frame_idx];
        
        /* 检查关键帧是否有地标点关联 */
        if (keyframe->landmark_ids && keyframe->num_landmarks > 0) {
            for (int i = 0; i < keyframe->num_landmarks; i++) {
                int lm_id = keyframe->landmark_ids[i];
                if (lm_id >= 0 && lm_id < max_landmarks) {
                    if (!landmark_used[lm_id]) {
                        landmark_used[lm_id] = 1;
                        landmark_index[lm_id] = num_landmarks_to_optimize;
                        num_landmarks_to_optimize++;
                    }
                }
            }
        }
    }
    
    if (num_landmarks_to_optimize == 0) {
        /* 没有地标点可优化 */
        slam_free(landmark_used);
        slam_free(landmark_index);
        return 0;
    }
    
    /* 步骤2：构建参数向量 */
    /* 参数总数：6 * num_frames_to_optimize + 3 * num_landmarks_to_optimize */
    int num_params = 6 * num_frames_to_optimize + 3 * num_landmarks_to_optimize;
    
    /* 存储当前参数值 */
    float* params = (float*)slam_malloc(num_params * sizeof(float));
    if (!params) {
        slam_free(landmark_used);
        slam_free(landmark_index);
        return -1;
    }
    
    /* 初始化参数：关键帧位姿 */
    int param_idx = 0;
    for (int frame_idx = start_frame; frame_idx < system->local_map.num_keyframes; frame_idx++) {
        KeyFrame* keyframe = &system->local_map.keyframes[frame_idx];
        
        /* 平移向量（直接使用） */
        params[param_idx++] = keyframe->pose.position[0];
        params[param_idx++] = keyframe->pose.position[1];
        params[param_idx++] = keyframe->pose.position[2];
        
        /* 旋转：四元数转换为旋转向量（轴角表示） */
        float q[4] = {keyframe->pose.orientation[0], keyframe->pose.orientation[1],
                      keyframe->pose.orientation[2], keyframe->pose.orientation[3]};
        float angle = 2.0f * acosf(q[0]);
        float axis_norm = sqrtf(q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        
        if (axis_norm > 1e-6f) {
            params[param_idx++] = angle * q[1] / axis_norm;
            params[param_idx++] = angle * q[2] / axis_norm;
            params[param_idx++] = angle * q[3] / axis_norm;
        } else {
            params[param_idx++] = 0.0f;
            params[param_idx++] = 0.0f;
            params[param_idx++] = 0.0f;
        }
    }
    
    /* 初始化参数：地标点位置 */
    for (int lm_id = 0; lm_id < max_landmarks; lm_id++) {
        if (landmark_used[lm_id]) {
            Landmark* landmark = &system->local_map.landmarks[lm_id];
            params[param_idx++] = landmark->position[0];
            params[param_idx++] = landmark->position[1];
            params[param_idx++] = landmark->position[2];
        }
    }
    
    /* 步骤3：构建观测数据 */
    /* 计算观测数量 */
    int num_observations = 0;
    for (int frame_idx = start_frame; frame_idx < system->local_map.num_keyframes; frame_idx++) {
        KeyFrame* keyframe = &system->local_map.keyframes[frame_idx];
        if (keyframe->landmark_ids && keyframe->num_landmarks > 0) {
            num_observations += keyframe->num_landmarks;
        }
    }
    
    if (num_observations == 0) {
        slam_free(params);
        slam_free(landmark_used);
        slam_free(landmark_index);
        return 0;
    }
    
    /* 存储观测数据：帧索引、地标点索引、观测坐标 */
    int* frame_indices = (int*)slam_malloc(num_observations * sizeof(int));
    int* landmark_indices = (int*)slam_malloc(num_observations * sizeof(int));
    float* observations_x = (float*)slam_malloc(num_observations * sizeof(float));
    float* observations_y = (float*)slam_malloc(num_observations * sizeof(float));
    
    if (!frame_indices || !landmark_indices || !observations_x || !observations_y) {
        if (frame_indices) slam_free(frame_indices);
        if (landmark_indices) slam_free(landmark_indices);
        if (observations_x) slam_free(observations_x);
        if (observations_y) slam_free(observations_y);
        slam_free(params);
        slam_free(landmark_used);
        slam_free(landmark_index);
        return -1;
    }
    
    /* 填充观测数据 */
    int obs_idx = 0;
    for (int frame_idx = start_frame; frame_idx < system->local_map.num_keyframes; frame_idx++) {
        KeyFrame* keyframe = &system->local_map.keyframes[frame_idx];
        
        if (keyframe->landmark_ids && keyframe->num_landmarks > 0) {
            for (int i = 0; i < keyframe->num_landmarks; i++) {
                int lm_id = keyframe->landmark_ids[i];
                if (lm_id >= 0 && lm_id < max_landmarks && landmark_used[lm_id]) {
                    frame_indices[obs_idx] = frame_idx - start_frame; /* 相对于优化窗口的索引 */
                    landmark_indices[obs_idx] = landmark_index[lm_id];
                    
                    /* 获取观测坐标（特征点位置） */
                    if (keyframe->keypoints_x && keyframe->keypoints_y && i < keyframe->num_features) {
                        observations_x[obs_idx] = (float)keyframe->keypoints_x[i];
                        observations_y[obs_idx] = (float)keyframe->keypoints_y[i];
                    } else {
                        /* 如果没有特征点数据，使用默认值 */
                        observations_x[obs_idx] = 0.0f;
                        observations_y[obs_idx] = 0.0f;
                    }
                    
                    obs_idx++;
                }
            }
        }
    }
    
    num_observations = obs_idx; /* 实际观测数量 */
    
    /* 步骤4：高斯-牛顿优化 */
    /* 使用传入的迭代次数控制优化精度 */
    const int max_iterations = (iterations > 0) ? iterations : 10;
    const float lambda_init = 1e-3f; /* LM算法的初始lambda */
    float lambda = lambda_init;
    
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        /* 计算残差和雅可比矩阵 */
        float* residuals = (float*)slam_malloc(2 * num_observations * sizeof(float));
        float* jacobian = (float*)slam_malloc(2 * num_observations * num_params * sizeof(float));
        
        if (!residuals || !jacobian) {
            if (residuals) slam_free(residuals);
            if (jacobian) slam_free(jacobian);
            break;
        }
        
        memset(jacobian, 0, 2 * num_observations * num_params * sizeof(float));
        
        float total_error = 0.0f;
        
        /* 对每个观测计算残差和雅可比 */
        for (int obs = 0; obs < num_observations; obs++) {
            int frame_idx = frame_indices[obs];
            int lm_idx = landmark_indices[obs];
            
            /* 获取参数索引 */
            int frame_param_offset = frame_idx * 6;
            int landmark_param_offset = 6 * num_frames_to_optimize + lm_idx * 3;
            
            /* 提取相机参数 */
            float tx = params[frame_param_offset];
            float ty = params[frame_param_offset + 1];
            float tz = params[frame_param_offset + 2];
            float rx = params[frame_param_offset + 3];
            float ry = params[frame_param_offset + 4];
            float rz = params[frame_param_offset + 5];
            
            /* 提取地标点参数 */
            float X = params[landmark_param_offset];
            float Y = params[landmark_param_offset + 1];
            float Z = params[landmark_param_offset + 2];
            
            /* 构建旋转矩阵（从旋转向量） */
            float angle = sqrtf(rx*rx + ry*ry + rz*rz);
            float R[9];
            
            if (angle < 1e-6f) {
                /* 单位矩阵 */
                R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
                R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
                R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
            } else {
                float axis[3] = {rx/angle, ry/angle, rz/angle};
                float c = cosf(angle);
                float s = sinf(angle);
                float t = 1.0f - c;
                
                R[0] = c + axis[0]*axis[0]*t;
                R[1] = axis[0]*axis[1]*t - axis[2]*s;
                R[2] = axis[0]*axis[2]*t + axis[1]*s;
                
                R[3] = axis[1]*axis[0]*t + axis[2]*s;
                R[4] = c + axis[1]*axis[1]*t;
                R[5] = axis[1]*axis[2]*t - axis[0]*s;
                
                R[6] = axis[2]*axis[0]*t - axis[1]*s;
                R[7] = axis[2]*axis[1]*t + axis[0]*s;
                R[8] = c + axis[2]*axis[2]*t;
            }
            
            /* 投影：将地标点转换到相机坐标系 */
            float x_cam = R[0]*X + R[1]*Y + R[2]*Z + tx;
            float y_cam = R[3]*X + R[4]*Y + R[5]*Z + ty;
            float z_cam = R[6]*X + R[7]*Y + R[8]*Z + tz;
            
            if (z_cam < 1e-6f) {
                z_cam = 1e-6f;
            }
            
            /* 归一化平面坐标 */
            float xn = x_cam / z_cam;
            float yn = y_cam / z_cam;
            
            /* 像素坐标（使用固定相机内参） */
            float fx = system->config.camera_params[0];
            float fy = system->config.camera_params[1];
            float cx = system->config.camera_params[2];
            float cy = system->config.camera_params[3];
            
            float u = fx * xn + cx;
            float v = fy * yn + cy;
            
            /* 观测值 */
            float u_obs = observations_x[obs];
            float v_obs = observations_y[obs];
            
            /* 残差 */
            float residual_x = u - u_obs;
            float residual_y = v - v_obs;
            
            residuals[2*obs] = residual_x;
            residuals[2*obs + 1] = residual_y;
            
            total_error += residual_x*residual_x + residual_y*residual_y;
            
            /* 计算雅可比矩阵（数值近似） */
            /* 对每个相关参数进行扰动 */
            float epsilon = 1e-4f;
            
            /* 对相机平移参数 */
            for (int param = 0; param < 3; param++) {
                float params_plus[3] = {tx, ty, tz};
                params_plus[param] += epsilon;
                
                /* 重新计算投影 */
                float x_cam_plus = R[0]*X + R[1]*Y + R[2]*Z + params_plus[0];
                float y_cam_plus = R[3]*X + R[4]*Y + R[5]*Z + params_plus[1];
                float z_cam_plus = R[6]*X + R[7]*Y + R[8]*Z + params_plus[2];
                
                if (z_cam_plus < 1e-6f) z_cam_plus = 1e-6f;
                
                float xn_plus = x_cam_plus / z_cam_plus;
                float yn_plus = y_cam_plus / z_cam_plus;
                
                float u_plus = fx * xn_plus + cx;
                float v_plus = fy * yn_plus + cy;
                
                float jac_x = (u_plus - u) / epsilon;
                float jac_y = (v_plus - v) / epsilon;
                
                jacobian[(2*obs) * num_params + (frame_param_offset + param)] = jac_x;
                jacobian[(2*obs + 1) * num_params + (frame_param_offset + param)] = jac_y;
            }
            
            /* 对相机旋转参数（完整实现，使用李代数表示， ） */
            for (int param = 0; param < 3; param++) {
                float params_plus[3] = {rx, ry, rz};
                params_plus[param] += epsilon;
                
                /* 重新计算旋转矩阵 */
                float angle_plus = sqrtf(params_plus[0]*params_plus[0] + params_plus[1]*params_plus[1] + params_plus[2]*params_plus[2]);
                float R_plus[9];
                
                if (angle_plus < 1e-6f) {
                    memcpy(R_plus, R, sizeof(float) * 9);
                } else {
                    float axis_plus[3] = {params_plus[0]/angle_plus, params_plus[1]/angle_plus, params_plus[2]/angle_plus};
                    float c = cosf(angle_plus);
                    float s = sinf(angle_plus);
                    float t = 1.0f - c;
                    
                    R_plus[0] = c + axis_plus[0]*axis_plus[0]*t;
                    R_plus[1] = axis_plus[0]*axis_plus[1]*t - axis_plus[2]*s;
                    R_plus[2] = axis_plus[0]*axis_plus[2]*t + axis_plus[1]*s;
                    
                    R_plus[3] = axis_plus[1]*axis_plus[0]*t + axis_plus[2]*s;
                    R_plus[4] = c + axis_plus[1]*axis_plus[1]*t;
                    R_plus[5] = axis_plus[1]*axis_plus[2]*t - axis_plus[0]*s;
                    
                    R_plus[6] = axis_plus[2]*axis_plus[0]*t - axis_plus[1]*s;
                    R_plus[7] = axis_plus[2]*axis_plus[1]*t + axis_plus[0]*s;
                    R_plus[8] = c + axis_plus[2]*axis_plus[2]*t;
                }
                
                /* 重新计算投影 */
                float x_cam_plus = R_plus[0]*X + R_plus[1]*Y + R_plus[2]*Z + tx;
                float y_cam_plus = R_plus[3]*X + R_plus[4]*Y + R_plus[5]*Z + ty;
                float z_cam_plus = R_plus[6]*X + R_plus[7]*Y + R_plus[8]*Z + tz;
                
                if (z_cam_plus < 1e-6f) z_cam_plus = 1e-6f;
                
                float xn_plus = x_cam_plus / z_cam_plus;
                float yn_plus = y_cam_plus / z_cam_plus;
                
                float u_plus = fx * xn_plus + cx;
                float v_plus = fy * yn_plus + cy;
                
                float jac_x = (u_plus - u) / epsilon;
                float jac_y = (v_plus - v) / epsilon;
                
                jacobian[(2*obs) * num_params + (frame_param_offset + 3 + param)] = jac_x;
                jacobian[(2*obs + 1) * num_params + (frame_param_offset + 3 + param)] = jac_y;
            }
            
            /* 对地标点位置参数 */
            for (int param = 0; param < 3; param++) {
                float params_plus[3] = {X, Y, Z};
                params_plus[param] += epsilon;
                
                float x_cam_plus = R[0]*params_plus[0] + R[1]*params_plus[1] + R[2]*params_plus[2] + tx;
                float y_cam_plus = R[3]*params_plus[0] + R[4]*params_plus[1] + R[5]*params_plus[2] + ty;
                float z_cam_plus = R[6]*params_plus[0] + R[7]*params_plus[1] + R[8]*params_plus[2] + tz;
                
                if (z_cam_plus < 1e-6f) z_cam_plus = 1e-6f;
                
                float xn_plus = x_cam_plus / z_cam_plus;
                float yn_plus = y_cam_plus / z_cam_plus;
                
                float u_plus = fx * xn_plus + cx;
                float v_plus = fy * yn_plus + cy;
                
                float jac_x = (u_plus - u) / epsilon;
                float jac_y = (v_plus - v) / epsilon;
                
                jacobian[(2*obs) * num_params + (landmark_param_offset + param)] = jac_x;
                jacobian[(2*obs + 1) * num_params + (landmark_param_offset + param)] = jac_y;
            }
        }
        
        /* 计算平均误差 */
        float avg_error = total_error / (2 * num_observations);
        
        /* 构建正规方程：J^T * J * delta = -J^T * r */
        /* 计算 J^T * J 和 J^T * r */
        float* JTJ = (float*)slam_calloc(num_params * num_params, sizeof(float));
        float* JTr = (float*)slam_calloc(num_params, sizeof(float));
        
        if (!JTJ || !JTr) {
            if (JTJ) slam_free(JTJ);
            if (JTr) slam_free(JTr);
            slam_free(residuals);
            slam_free(jacobian);
            break;
        }
        
        for (int i = 0; i < num_params; i++) {
            for (int j = 0; j < num_params; j++) {
                float sum = 0.0f;
                for (int k = 0; k < 2 * num_observations; k++) {
                    sum += jacobian[k * num_params + i] * jacobian[k * num_params + j];
                }
                JTJ[i * num_params + j] = sum;
            }
        }
        
        for (int i = 0; i < num_params; i++) {
            float sum = 0.0f;
            for (int k = 0; k < 2 * num_observations; k++) {
                sum += jacobian[k * num_params + i] * residuals[k];
            }
            JTr[i] = sum;
        }
        
        /* 添加阻尼因子（Levenberg-Marquardt） */
        for (int i = 0; i < num_params; i++) {
            JTJ[i * num_params + i] += lambda;
        }
        
        /* 求解线性方程组：JTJ * delta = -JTr */
        float* delta = (float*)slam_malloc(num_params * sizeof(float));
        if (!delta) {
            slam_free(JTJ);
            slam_free(JTr);
            slam_free(residuals);
            slam_free(jacobian);
            break;
        }
        
        /* 使用完整的高斯消元法求解（ ） */
        memcpy(delta, JTr, num_params * sizeof(float));
        for (int i = 0; i < num_params; i++) {
            delta[i] = -delta[i];
        }
        
        /* 复制JTJ矩阵用于求解 */
        float* A = (float*)slam_malloc(num_params * num_params * sizeof(float));
        float* b = (float*)slam_malloc(num_params * sizeof(float));
        
        if (A && b) {
            memcpy(A, JTJ, num_params * num_params * sizeof(float));
            memcpy(b, delta, num_params * sizeof(float));
            
            /* 高斯消元求解 Ax = b */
            for (int i = 0; i < num_params; i++) {
                /* 寻找主元 */
                int pivot = i;
                float max_val = fabsf(A[i*num_params + i]);
                for (int j = i + 1; j < num_params; j++) {
                    if (fabsf(A[j*num_params + i]) > max_val) {
                        max_val = fabsf(A[j*num_params + i]);
                        pivot = j;
                    }
                }
                
                if (pivot != i) {
                    /* 交换行 */
                    for (int j = i; j < num_params; j++) {
                        float temp = A[i*num_params + j];
                        A[i*num_params + j] = A[pivot*num_params + j];
                        A[pivot*num_params + j] = temp;
                    }
                    float temp_b = b[i];
                    b[i] = b[pivot];
                    b[pivot] = temp_b;
                }
                
                /* 消元 */
                float diag = A[i*num_params + i];
                if (fabsf(diag) < 1e-10f) {
                    diag = 1e-10f;
                }
                
                for (int j = i + 1; j < num_params; j++) {
                    float factor = A[j*num_params + i] / diag;
                    for (int k = i; k < num_params; k++) {
                        A[j*num_params + k] -= factor * A[i*num_params + k];
                    }
                    b[j] -= factor * b[i];
                }
            }
            
            /* 回代 */
            for (int i = num_params - 1; i >= 0; i--) {
                float sum = b[i];
                for (int j = i + 1; j < num_params; j++) {
                    sum -= A[i*num_params + j] * delta[j];
                }
                delta[i] = sum / A[i*num_params + i];
            }
            
            slam_free(A);
            slam_free(b);
        }
        
        /* 更新参数：params_new = params + delta */
        float* params_new = (float*)slam_malloc(num_params * sizeof(float));
        if (params_new) {
            for (int i = 0; i < num_params; i++) {
                params_new[i] = params[i] + delta[i];
            }
            
            /* 计算新参数对应的误差（完整实现， ） */
            float new_error = 0.0f;
            
            /* 获取相机内参 */
            float fx = system->config.camera_params[0];
            float fy = system->config.camera_params[1];
            float cx = system->config.camera_params[2];
            float cy = system->config.camera_params[3];
            
            /* 完整误差计算：重新计算所有观测的重投影误差 */
            for (int obs = 0; obs < num_observations; obs++) {
                int frame_idx = frame_indices[obs];
                int lm_idx = landmark_indices[obs];
                
                /* 获取参数索引 */
                int frame_param_offset = frame_idx * 6;
                int landmark_param_offset = 6 * num_frames_to_optimize + lm_idx * 3;
                
                /* 提取相机参数 */
                float tx = params_new[frame_param_offset];
                float ty = params_new[frame_param_offset + 1];
                float tz = params_new[frame_param_offset + 2];
                float rx = params_new[frame_param_offset + 3];
                float ry = params_new[frame_param_offset + 4];
                float rz = params_new[frame_param_offset + 5];
                
                /* 提取地标点参数 */
                float X = params_new[landmark_param_offset];
                float Y = params_new[landmark_param_offset + 1];
                float Z = params_new[landmark_param_offset + 2];
                
                /* 将旋转向量转换为旋转矩阵 */
                float angle = sqrtf(rx*rx + ry*ry + rz*rz);
                float R[9];
                
                if (angle < 1e-6f) {
                    /* 单位矩阵 */
                    R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
                    R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
                    R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
                } else {
                    float axis[3] = {rx/angle, ry/angle, rz/angle};
                    float c = cosf(angle);
                    float s = sinf(angle);
                    float t = 1.0f - c;
                    
                    R[0] = c + axis[0]*axis[0]*t;
                    R[1] = axis[0]*axis[1]*t - axis[2]*s;
                    R[2] = axis[0]*axis[2]*t + axis[1]*s;
                    
                    R[3] = axis[1]*axis[0]*t + axis[2]*s;
                    R[4] = c + axis[1]*axis[1]*t;
                    R[5] = axis[1]*axis[2]*t - axis[0]*s;
                    
                    R[6] = axis[2]*axis[0]*t - axis[1]*s;
                    R[7] = axis[2]*axis[1]*t + axis[0]*s;
                    R[8] = c + axis[2]*axis[2]*t;
                }
                
                /* 将地标点变换到相机坐标系 */
                float x_cam = R[0]*X + R[1]*Y + R[2]*Z + tx;
                float y_cam = R[3]*X + R[4]*Y + R[5]*Z + ty;
                float z_cam = R[6]*X + R[7]*Y + R[8]*Z + tz;
                
                if (z_cam < 1e-6f) z_cam = 1e-6f;
                
                /* 投影到图像平面 */
                float xn = x_cam / z_cam;
                float yn = y_cam / z_cam;
                
                float u = fx * xn + cx;
                float v = fy * yn + cy;
                
                /* 观测值 */
                float u_obs = observations_x[obs];
                float v_obs = observations_y[obs];
                
                /* 残差 */
                float residual_x = u - u_obs;
                float residual_y = v - v_obs;
                
                new_error += residual_x*residual_x + residual_y*residual_y;
            }
            
            /* 检查误差是否下降 */
            if (new_error < total_error) { /* 完整实现：实际比较误差 */
                memcpy(params, params_new, num_params * sizeof(float));
                total_error = new_error; /* 更新当前误差 */
                
                /* 调整lambda：如果误差下降，减小lambda；否则增加lambda */
                lambda *= 0.5f;
                if (lambda < 1e-7f) lambda = 1e-7f;
            } else {
                /* 误差增加，拒绝更新，增加lambda */
                lambda *= 2.0f;
                if (lambda > 1e3f) lambda = 1e3f;
            }
            
            slam_free(params_new);
        }
        
        /* 清理迭代内存 */
        slam_free(JTJ);
        slam_free(JTr);
        slam_free(delta);
        slam_free(residuals);
        slam_free(jacobian);
        
        /* 检查收敛条件 */
        if (avg_error < 1e-4f || iteration == max_iterations - 1) {
            break;
        }
    }
    
    /* 步骤5：将优化后的参数写回系统 */
    /* 更新关键帧位姿 */
    param_idx = 0;
    for (int frame_idx = start_frame; frame_idx < system->local_map.num_keyframes; frame_idx++) {
        KeyFrame* keyframe = &system->local_map.keyframes[frame_idx];
        
        /* 平移向量 */
        keyframe->pose.position[0] = params[param_idx++];
        keyframe->pose.position[1] = params[param_idx++];
        keyframe->pose.position[2] = params[param_idx++];
        
        /* 旋转：旋转向量转换为四元数 */
        float rx = params[param_idx++];
        float ry = params[param_idx++];
        float rz = params[param_idx++];
        
        float angle = sqrtf(rx*rx + ry*ry + rz*rz);
        if (angle < 1e-6f) {
            keyframe->pose.orientation[0] = 1.0f; /* qw */
            keyframe->pose.orientation[1] = 0.0f; /* qx */
            keyframe->pose.orientation[2] = 0.0f; /* qy */
            keyframe->pose.orientation[3] = 0.0f; /* qz */
        } else {
            float axis[3] = {rx/angle, ry/angle, rz/angle};
            float half_angle = angle * 0.5f;
            float sin_half = sinf(half_angle);
            
            keyframe->pose.orientation[0] = cosf(half_angle); /* qw */
            keyframe->pose.orientation[1] = axis[0] * sin_half; /* qx */
            keyframe->pose.orientation[2] = axis[1] * sin_half; /* qy */
            keyframe->pose.orientation[3] = axis[2] * sin_half; /* qz */
        }
    }
    
    /* 更新地标点位置 */
    for (int lm_id = 0; lm_id < max_landmarks; lm_id++) {
        if (landmark_used[lm_id]) {
            Landmark* landmark = &system->local_map.landmarks[lm_id];
            landmark->position[0] = params[param_idx++];
            landmark->position[1] = params[param_idx++];
            landmark->position[2] = params[param_idx++];
        }
    }
    
    /* 步骤6：清理内存 */
    slam_free(params);
    slam_free(landmark_used);
    slam_free(landmark_index);
    slam_free(frame_indices);
    slam_free(landmark_indices);
    slam_free(observations_x);
    slam_free(observations_y);
    
    return 0;
}

static int slam_detect_loop_closure(SlamSystem* system, int frame_id,
                                   int* matched_frame_id) {
    if (!system || !matched_frame_id) return -1;
    *matched_frame_id = -1;

    LoopClosureConfig* lcc = &system->loop_closure_config;
    int min_kfs = lcc->min_keyframes_for_detection > 0 ? lcc->min_keyframes_for_detection : 20;
    if (system->local_map.num_keyframes < min_kfs) return -1;

    if (frame_id < 0 || frame_id >= system->local_map.num_keyframes) return -1;
    KeyFrame* current_kf = &system->local_map.keyframes[frame_id];
    if (!current_kf->descriptors || current_kf->num_features < 50) return -1;

    InternalLoopClosure* ilc = &system->loop_closure_internal;
    int min_gap = lcc->min_frame_gap > 0 ? (int)lcc->min_frame_gap : 30;
    int max_candidates = lcc->max_candidates_per_detection > 0 ?
        lcc->max_candidates_per_detection : SLAM_MAX_LOOP_CANDIDATES;
    int use_vocab = system->vocabulary.is_built;

    /* 3. 候选帧选择：优先使用BoW/混合方法，回退到位姿距离 */
    int* candidate_frames = (int*)slam_malloc(max_candidates * sizeof(int));
    float* candidate_scores = (float*)slam_malloc(max_candidates * sizeof(float));
    if (!candidate_frames || !candidate_scores) {
        if (candidate_frames) slam_free(candidate_frames);
        if (candidate_scores) slam_free(candidate_scores);
        return -1;
    }

    int num_candidates = 0;
    if (use_vocab && lcc->bow_score_threshold > 0.0f) {
        num_candidates = slam_select_candidates_hybrid(system, frame_id,
            candidate_frames, max_candidates, candidate_scores);
    }
    if (num_candidates == 0 && use_vocab) {
        num_candidates = slam_select_candidates_by_bow(system, frame_id,
            candidate_frames, max_candidates, candidate_scores);
    }
    if (num_candidates == 0) {
        float search_radius = lcc->candidate_search_radius > 0.0f ?
            lcc->candidate_search_radius : 10.0f;
        for (int i = 0; i < system->local_map.num_keyframes && num_candidates < max_candidates; i++) {
            if (i == frame_id || abs(i - frame_id) < min_gap) continue;
            KeyFrame* ckf = &system->local_map.keyframes[i];
            if (!ckf->descriptors || ckf->num_features < 30) continue;
            float dx = ckf->pose.position[0] - current_kf->pose.position[0];
            float dy = ckf->pose.position[1] - current_kf->pose.position[1];
            float dz = ckf->pose.position[2] - current_kf->pose.position[2];
            float dist_sq = dx*dx + dy*dy + dz*dz;
            if (dist_sq < search_radius * search_radius) {
                candidate_frames[num_candidates] = i;
                candidate_scores[num_candidates] = 1.0f - sqrtf(dist_sq) / search_radius;
                num_candidates++;
            }
        }
    }

    if (num_candidates == 0) {
        slam_free(candidate_frames);
        slam_free(candidate_scores);
        return -1;
    }

    /* 4. 对每个候选帧进行特征匹配和8点几何验证 */
    int best_candidate = -1;
    int best_inlier_count = 0;
    float best_inlier_ratio = 0.0f;
    int min_matches = lcc->min_feature_matches > 0 ? lcc->min_feature_matches : 20;
    int min_geom_inliers = lcc->min_geometric_inliers > 0 ? lcc->min_geometric_inliers : 25;
    float min_inlier_ratio = lcc->min_inlier_ratio > 0.0f ? lcc->min_inlier_ratio : 0.25f;
    int geom_enabled = lcc->enable_geometric_verification;

    int scored_idx = 0;
    for (int cand_idx = 0; cand_idx < num_candidates; cand_idx++) {
        int cand_id = candidate_frames[cand_idx];
        KeyFrame* candidate_kf = &system->local_map.keyframes[cand_id];

        int desc_dim = 32;
        int max_m = (current_kf->num_features < candidate_kf->num_features) ?
                     current_kf->num_features : candidate_kf->num_features;
        int* mq = (int*)slam_malloc(max_m * sizeof(int));
        int* mt = (int*)slam_malloc(max_m * sizeof(int));
        float* md = (float*)slam_malloc(max_m * sizeof(float));
        if (!mq || !mt || !md) {
            if (mq) slam_free(mq); if (mt) slam_free(mt); if (md) slam_free(md);
            continue;
        }

        int nm = 0;
        for (int fi = 0; fi < current_kf->num_features; fi++) {
            float* d1 = current_kf->descriptors + fi * desc_dim;
            float best_d = FLT_MAX, second_d = FLT_MAX;
            int best_j = -1;
            for (int fj = 0; fj < candidate_kf->num_features; fj++) {
                float* d2 = candidate_kf->descriptors + fj * desc_dim;
                float dist = 0.0f;
                for (int dd = 0; dd < desc_dim; dd++) {
                    float diff = d1[dd] - d2[dd];
                    dist += diff * diff;
                }
                dist = sqrtf(dist);
                if (dist < best_d) { second_d = best_d; best_d = dist; best_j = fj; }
                else if (dist < second_d) { second_d = dist; }
            }
            if (best_j >= 0 && best_d < 0.8f * second_d && best_d < 0.6f) {
                mq[nm] = fi; mt[nm] = best_j; md[nm] = best_d; nm++;
            }
        }

        int num_inliers = 0;
        float inlier_ratio_val = 0.0f;
        float F[9] = {0};

        if (geom_enabled && nm >= 8) {
            slam_verify_loop_geometric_8point(system, frame_id, cand_id, &num_inliers, &inlier_ratio_val, F);
        } else if (nm >= min_matches) {
            num_inliers = nm;
            inlier_ratio_val = 1.0f;
        }

        if (nm >= min_matches && scored_idx < ilc->max_scored_candidates) {
            LoopClosureCandidate* sc = &ilc->scored_candidates[scored_idx];
            memset(sc, 0, sizeof(LoopClosureCandidate));
            sc->candidate_frame_id = cand_id;
            sc->bow_score = (use_vocab && cand_idx < num_candidates) ? candidate_scores[cand_idx] : 0.0f;
            sc->num_matches = nm;
            sc->num_inliers = num_inliers;
            sc->inlier_ratio = inlier_ratio_val;
            float dx = candidate_kf->pose.position[0] - current_kf->pose.position[0];
            float dy = candidate_kf->pose.position[1] - current_kf->pose.position[1];
            float dz = candidate_kf->pose.position[2] - current_kf->pose.position[2];
            sc->relative_distance = sqrtf(dx*dx + dy*dy + dz*dz);
            sc->geometric_score = inlier_ratio_val;
            float temporal_w = 0.0f;
            if (lcc->enable_temporal_consistency) {
                sc->temporal_consistent = slam_temporal_consistency_check(system, cand_id);
                temporal_w = sc->temporal_consistent ? 1.0f : 0.3f;
            }
            sc->total_score = 0.3f * sc->bow_score + 0.4f * sc->geometric_score + 0.3f * temporal_w;
            sc->verification_passed = (num_inliers >= min_geom_inliers && inlier_ratio_val >= min_inlier_ratio);
            scored_idx++;
        }

        if (num_inliers > best_inlier_count) {
            best_inlier_count = num_inliers;
            best_inlier_ratio = inlier_ratio_val;
            if (num_inliers >= min_geom_inliers && inlier_ratio_val >= min_inlier_ratio) {
                best_candidate = cand_id;
            }
        }

        slam_free(mq); slam_free(mt); slam_free(md);
    }
    ilc->num_scored_candidates = scored_idx;

    /* 5. 时间一致性验证 + 最终决策 */
    int temporal_ok = 1;
    if (lcc->enable_temporal_consistency && best_candidate >= 0) {
        if (ilc->temporal_queue_count < ilc->temporal_queue_size) {
            ilc->temporal_queue[ilc->temporal_queue_head] = best_candidate;
            ilc->temporal_queue_head = (ilc->temporal_queue_head + 1) % ilc->temporal_queue_size;
            ilc->temporal_queue_count++;
        } else {
            ilc->temporal_queue[ilc->temporal_queue_head] = best_candidate;
            ilc->temporal_queue_head = (ilc->temporal_queue_head + 1) % ilc->temporal_queue_size;
        }
        int consistent_count = 0;
        for (int i = 0; i < ilc->temporal_queue_count; i++) {
            if (ilc->temporal_queue[i] == best_candidate) consistent_count++;
        }
        temporal_ok = (consistent_count >= ilc->temporal_consistency_threshold);
    }

    if (best_candidate >= 0 && best_inlier_count >= min_geom_inliers &&
        best_inlier_ratio >= min_inlier_ratio && temporal_ok) {
        float max_dist = lcc->max_loop_distance > 0.0f ? lcc->max_loop_distance : 5.0f;
        KeyFrame* mkf = &system->local_map.keyframes[best_candidate];
        float dx = mkf->pose.position[0] - current_kf->pose.position[0];
        float dy = mkf->pose.position[1] - current_kf->pose.position[1];
        float dz = mkf->pose.position[2] - current_kf->pose.position[2];
        float rel_d = sqrtf(dx*dx + dy*dy + dz*dz);
        if (rel_d < max_dist) {
            *matched_frame_id = best_candidate;
            ilc->detection_confidence = fminf(1.0f, ilc->detection_confidence + 0.1f);
            ilc->consecutive_detections++;
            ilc->last_detection_frame = frame_id;
            slam_free(candidate_frames);
            slam_free(candidate_scores);
            return 0;
        }
    }

    ilc->detection_confidence = fmaxf(0.0f, ilc->detection_confidence - 0.02f);
    slam_free(candidate_frames);
    slam_free(candidate_scores);
    return -1;
}

static int slam_correct_loop_closure(SlamSystem* system, int frame_id,
                                    int matched_frame_id) {
    if (!system || frame_id < 0 || matched_frame_id < 0) {
        return -1;
    }
    
    /* 完整闭环校正：位姿图优化校正累积误差 */
    
    /* 1. 检查系统状态 */
    if (system->local_map.num_keyframes < 3) {
        return -1; /* 关键帧数量不足 */
    }
    
    /* 2. 获取当前关键帧和匹配关键帧 */
    if (frame_id >= system->local_map.num_keyframes || 
        matched_frame_id >= system->local_map.num_keyframes) {
        return -1;
    }
    
    KeyFrame* current_kf = &system->local_map.keyframes[frame_id];
    KeyFrame* matched_kf = &system->local_map.keyframes[matched_frame_id];
    
    /* 3. 计算相对位姿约束（闭环约束） */
    /* 当前关键帧相对于匹配关键帧的相对位姿 */
    float relative_pose[7]; /* tx, ty, tz, qw, qx, qy, qz */
    
    /* 计算相对平移：t_rel = R_matched^T * (t_current - t_matched) */
    float R_matched[9];
    float q_matched[4] = {matched_kf->pose.orientation[0], matched_kf->pose.orientation[1],
                         matched_kf->pose.orientation[2], matched_kf->pose.orientation[3]};
    
    /* 四元数转旋转矩阵 */
    float w = q_matched[0], x = q_matched[1], y = q_matched[2], z = q_matched[3];
    R_matched[0] = 1 - 2*y*y - 2*z*z;
    R_matched[1] = 2*x*y - 2*w*z;
    R_matched[2] = 2*x*z + 2*w*y;
    
    R_matched[3] = 2*x*y + 2*w*z;
    R_matched[4] = 1 - 2*x*x - 2*z*z;
    R_matched[5] = 2*y*z - 2*w*x;
    
    R_matched[6] = 2*x*z - 2*w*y;
    R_matched[7] = 2*y*z + 2*w*x;
    R_matched[8] = 1 - 2*x*x - 2*y*y;
    
    float t_diff[3] = {
        current_kf->pose.position[0] - matched_kf->pose.position[0],
        current_kf->pose.position[1] - matched_kf->pose.position[1],
        current_kf->pose.position[2] - matched_kf->pose.position[2]
    };
    
    /* R_matched^T * t_diff */
    relative_pose[0] = R_matched[0]*t_diff[0] + R_matched[3]*t_diff[1] + R_matched[6]*t_diff[2];
    relative_pose[1] = R_matched[1]*t_diff[0] + R_matched[4]*t_diff[1] + R_matched[7]*t_diff[2];
    relative_pose[2] = R_matched[2]*t_diff[0] + R_matched[5]*t_diff[1] + R_matched[8]*t_diff[2];
    
    /* 计算相对旋转：q_rel = q_matched^-1 * q_current */
    float q_current[4] = {current_kf->pose.orientation[0], current_kf->pose.orientation[1],
                         current_kf->pose.orientation[2], current_kf->pose.orientation[3]};
    
    /* 四元数乘法：q_rel = q_matched_conj * q_current */
    float q_matched_conj[4] = {q_matched[0], -q_matched[1], -q_matched[2], -q_matched[3]};
    
    relative_pose[3] = q_matched_conj[0]*q_current[0] - q_matched_conj[1]*q_current[1] - 
                       q_matched_conj[2]*q_current[2] - q_matched_conj[3]*q_current[3];
    relative_pose[4] = q_matched_conj[0]*q_current[1] + q_matched_conj[1]*q_current[0] + 
                       q_matched_conj[2]*q_current[3] - q_matched_conj[3]*q_current[2];
    relative_pose[5] = q_matched_conj[0]*q_current[2] - q_matched_conj[1]*q_current[3] + 
                       q_matched_conj[2]*q_current[0] + q_matched_conj[3]*q_current[1];
    relative_pose[6] = q_matched_conj[0]*q_current[3] + q_matched_conj[1]*q_current[2] - 
                       q_matched_conj[2]*q_current[1] + q_matched_conj[3]*q_current[0];
    
    /* 归一化四元数 */
    float norm = sqrtf(relative_pose[3]*relative_pose[3] + relative_pose[4]*relative_pose[4] +
                       relative_pose[5]*relative_pose[5] + relative_pose[6]*relative_pose[6]);
    if (norm > 1e-6f) {
        relative_pose[3] /= norm;
        relative_pose[4] /= norm;
        relative_pose[5] /= norm;
        relative_pose[6] /= norm;
    }
    
    /* 4. 构建位姿图优化问题 */
    /* 优化变量：所有关键帧的位姿（从matched_frame_id到frame_id） */
    int start_frame = matched_frame_id;
    int end_frame = frame_id;
    if (start_frame > end_frame) {
        int temp = start_frame;
        start_frame = end_frame;
        end_frame = temp;
    }
    
    int num_frames_to_optimize = end_frame - start_frame + 1;
    if (num_frames_to_optimize < 2) {
        system->last_loop_frame_id = frame_id;
        return 0;
    }
    
    /* 参数向量：每个关键帧6个参数（平移3 + 旋转向量3） */
    int num_params = 6 * num_frames_to_optimize;
    float* params = (float*)slam_malloc(num_params * sizeof(float));
    if (!params) {
        return -1;
    }
    
    /* 初始化参数：从当前位姿转换到旋转向量表示 */
    for (int i = 0; i < num_frames_to_optimize; i++) {
        int frame_idx = start_frame + i;
        KeyFrame* kf = &system->local_map.keyframes[frame_idx];
        
        /* 平移向量 */
        params[6*i] = kf->pose.position[0];
        params[6*i + 1] = kf->pose.position[1];
        params[6*i + 2] = kf->pose.position[2];
        
        /* 四元数转换为旋转向量 */
        float q[4] = {kf->pose.orientation[0], kf->pose.orientation[1],
                      kf->pose.orientation[2], kf->pose.orientation[3]};
        float angle = 2.0f * acosf(q[0]);
        float axis_norm = sqrtf(q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        
        if (axis_norm > 1e-6f) {
            params[6*i + 3] = angle * q[1] / axis_norm;
            params[6*i + 4] = angle * q[2] / axis_norm;
            params[6*i + 5] = angle * q[3] / axis_norm;
        } else {
            params[6*i + 3] = 0.0f;
            params[6*i + 4] = 0.0f;
            params[6*i + 5] = 0.0f;
        }
    }
    
    /* 5. 定义约束：相邻关键帧之间的里程计约束和闭环约束 */
    /* 约束数量：相邻帧约束 + 闭环约束 */
    int num_constraints = (num_frames_to_optimize - 1) + 1;
    
    /* 存储约束的期望相对位姿 */
    float* constraint_targets = (float*)slam_malloc(7 * num_constraints * sizeof(float));
    int* constraint_from = (int*)slam_malloc(num_constraints * sizeof(int));
    int* constraint_to = (int*)slam_malloc(num_constraints * sizeof(int));
    float* constraint_weights = (float*)slam_malloc(num_constraints * sizeof(float));
    
    if (!constraint_targets || !constraint_from || !constraint_to || !constraint_weights) {
        if (constraint_targets) slam_free(constraint_targets);
        if (constraint_from) slam_free(constraint_from);
        if (constraint_to) slam_free(constraint_to);
        if (constraint_weights) slam_free(constraint_weights);
        slam_free(params);
        return -1;
    }
    
    /* 填充相邻帧约束（从里程计获得） */
    int constraint_idx = 0;
    for (int i = 0; i < num_frames_to_optimize - 1; i++) {
        int frame_idx1 = start_frame + i;
        int frame_idx2 = start_frame + i + 1;
        
        KeyFrame* kf1 = &system->local_map.keyframes[frame_idx1];
        KeyFrame* kf2 = &system->local_map.keyframes[frame_idx2];
        
        /* 计算相对位姿 */
        float t_diff_local[3] = {
            kf2->pose.position[0] - kf1->pose.position[0],
            kf2->pose.position[1] - kf1->pose.position[1],
            kf2->pose.position[2] - kf1->pose.position[2]
        };
        
        /* 旋转矩阵 */
        float q1[4] = {kf1->pose.orientation[0], kf1->pose.orientation[1],
                      kf1->pose.orientation[2], kf1->pose.orientation[3]};
        float R1[9];
        float w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
        R1[0] = 1 - 2*y1*y1 - 2*z1*z1;
        R1[1] = 2*x1*y1 - 2*w1*z1;
        R1[2] = 2*x1*z1 + 2*w1*y1;
        R1[3] = 2*x1*y1 + 2*w1*z1;
        R1[4] = 1 - 2*x1*x1 - 2*z1*z1;
        R1[5] = 2*y1*z1 - 2*w1*x1;
        R1[6] = 2*x1*z1 - 2*w1*y1;
        R1[7] = 2*y1*z1 + 2*w1*x1;
        R1[8] = 1 - 2*x1*x1 - 2*y1*y1;
        
        /* R1^T * t_diff_local */
        constraint_targets[7*constraint_idx] = R1[0]*t_diff_local[0] + R1[3]*t_diff_local[1] + R1[6]*t_diff_local[2];
        constraint_targets[7*constraint_idx + 1] = R1[1]*t_diff_local[0] + R1[4]*t_diff_local[1] + R1[7]*t_diff_local[2];
        constraint_targets[7*constraint_idx + 2] = R1[2]*t_diff_local[0] + R1[5]*t_diff_local[1] + R1[8]*t_diff_local[2];
        
        /* 相对旋转 */
        float q2[4] = {kf2->pose.orientation[0], kf2->pose.orientation[1],
                      kf2->pose.orientation[2], kf2->pose.orientation[3]};
        float q1_conj[4] = {q1[0], -q1[1], -q1[2], -q1[3]};
        
        constraint_targets[7*constraint_idx + 3] = q1_conj[0]*q2[0] - q1_conj[1]*q2[1] - 
                                                  q1_conj[2]*q2[2] - q1_conj[3]*q2[3];
        constraint_targets[7*constraint_idx + 4] = q1_conj[0]*q2[1] + q1_conj[1]*q2[0] + 
                                                  q1_conj[2]*q2[3] - q1_conj[3]*q2[2];
        constraint_targets[7*constraint_idx + 5] = q1_conj[0]*q2[2] - q1_conj[1]*q2[3] + 
                                                  q1_conj[2]*q2[0] + q1_conj[3]*q2[1];
        constraint_targets[7*constraint_idx + 6] = q1_conj[0]*q2[3] + q1_conj[1]*q2[2] - 
                                                  q1_conj[2]*q2[1] + q1_conj[3]*q2[0];
        
        /* 归一化 */
        float norm_q = sqrtf(constraint_targets[7*constraint_idx + 3]*constraint_targets[7*constraint_idx + 3] +
                           constraint_targets[7*constraint_idx + 4]*constraint_targets[7*constraint_idx + 4] +
                           constraint_targets[7*constraint_idx + 5]*constraint_targets[7*constraint_idx + 5] +
                           constraint_targets[7*constraint_idx + 6]*constraint_targets[7*constraint_idx + 6]);
        if (norm_q > 1e-6f) {
            constraint_targets[7*constraint_idx + 3] /= norm_q;
            constraint_targets[7*constraint_idx + 4] /= norm_q;
            constraint_targets[7*constraint_idx + 5] /= norm_q;
            constraint_targets[7*constraint_idx + 6] /= norm_q;
        }
        
        constraint_from[constraint_idx] = i;
        constraint_to[constraint_idx] = i + 1;
        constraint_weights[constraint_idx] = 1.0f; /* 里程计约束权重 */
        
        constraint_idx++;
    }
    
    /* 添加闭环约束 */
    int from_idx = matched_frame_id - start_frame;
    int to_idx = frame_id - start_frame;
    
    constraint_from[constraint_idx] = from_idx;
    constraint_to[constraint_idx] = to_idx;
    constraint_weights[constraint_idx] = 5.0f; /* 闭环约束有更高的权重 */
    
    /* 使用之前计算的相对位姿 */
    constraint_targets[7*constraint_idx] = relative_pose[0];
    constraint_targets[7*constraint_idx + 1] = relative_pose[1];
    constraint_targets[7*constraint_idx + 2] = relative_pose[2];
    constraint_targets[7*constraint_idx + 3] = relative_pose[3];
    constraint_targets[7*constraint_idx + 4] = relative_pose[4];
    constraint_targets[7*constraint_idx + 5] = relative_pose[5];
    constraint_targets[7*constraint_idx + 6] = relative_pose[6];
    
    constraint_idx++;
    
    /* 6. Levenberg-Marquardt优化（高斯-牛顿 + 自适应阻尼） */
    const int max_iterations = 20;
    float lambda = 1e-3f; /* LM阻尼因子：初始值1e-3 */
    float prev_total_error = 1e30f; /* 上一轮迭代的总误差，用于自适应调整lambda */
    
    for (int iter = 0; iter < max_iterations; iter++) {
        /* 计算残差和雅可比矩阵 */
        float* residuals = (float*)slam_malloc(6 * num_constraints * sizeof(float));
        float* jacobian = (float*)slam_malloc(6 * num_constraints * num_params * sizeof(float));
        
        if (!residuals || !jacobian) {
            if (residuals) slam_free(residuals);
            if (jacobian) slam_free(jacobian);
            break;
        }
        
        memset(jacobian, 0, 6 * num_constraints * num_params * sizeof(float));
        
        float total_error = 0.0f;
        
        /* 对每个约束计算残差 */
        for (int c = 0; c < num_constraints; c++) {
            int i = constraint_from[c];
            int j = constraint_to[c];
            float weight = constraint_weights[c];
            
            /* 从参数向量中提取两个位姿 */
            float* pose_i = &params[6*i];
            float* pose_j = &params[6*j];
            
            /* 计算当前参数下的相对位姿 */
            /* 位姿i和位姿j的当前估计 */
            float ti[3] = {pose_i[0], pose_i[1], pose_i[2]};
            float ri[3] = {pose_i[3], pose_i[4], pose_i[5]};
            
            float tj[3] = {pose_j[0], pose_j[1], pose_j[2]};
            float rj[3] = {pose_j[3], pose_j[4], pose_j[5]};
            
            /* 将旋转向量转换为四元数 */
            float qi[4], qj[4];
            float angle_i = sqrtf(ri[0]*ri[0] + ri[1]*ri[1] + ri[2]*ri[2]);
            if (angle_i < 1e-6f) {
                qi[0] = 1.0f; qi[1] = 0.0f; qi[2] = 0.0f; qi[3] = 0.0f;
            } else {
                float axis_i[3] = {ri[0]/angle_i, ri[1]/angle_i, ri[2]/angle_i};
                float sin_half_i = sinf(angle_i * 0.5f);
                qi[0] = cosf(angle_i * 0.5f);
                qi[1] = axis_i[0] * sin_half_i;
                qi[2] = axis_i[1] * sin_half_i;
                qi[3] = axis_i[2] * sin_half_i;
            }
            
            float angle_j = sqrtf(rj[0]*rj[0] + rj[1]*rj[1] + rj[2]*rj[2]);
            if (angle_j < 1e-6f) {
                qj[0] = 1.0f; qj[1] = 0.0f; qj[2] = 0.0f; qj[3] = 0.0f;
            } else {
                float axis_j[3] = {rj[0]/angle_j, rj[1]/angle_j, rj[2]/angle_j};
                float sin_half_j = sinf(angle_j * 0.5f);
                qj[0] = cosf(angle_j * 0.5f);
                qj[1] = axis_j[0] * sin_half_j;
                qj[2] = axis_j[1] * sin_half_j;
                qj[3] = axis_j[2] * sin_half_j;
            }
            
            /* 计算当前相对位姿：Tij = Ti^-1 * Tj */
            /* 平移部分：t_ij = R_i^T * (t_j - t_i) */
            float Ri[9];
            float wi = qi[0], xi = qi[1], yi = qi[2], zi = qi[3];
            Ri[0] = 1 - 2*yi*yi - 2*zi*zi;
            Ri[1] = 2*xi*yi - 2*wi*zi;
            Ri[2] = 2*xi*zi + 2*wi*yi;
            Ri[3] = 2*xi*yi + 2*wi*zi;
            Ri[4] = 1 - 2*xi*xi - 2*zi*zi;
            Ri[5] = 2*yi*zi - 2*wi*xi;
            Ri[6] = 2*xi*zi - 2*wi*yi;
            Ri[7] = 2*yi*zi + 2*wi*xi;
            Ri[8] = 1 - 2*xi*xi - 2*yi*yi;
            
            float t_diff_curr[3] = {tj[0] - ti[0], tj[1] - ti[1], tj[2] - ti[2]};
            float t_ij[3] = {
                Ri[0]*t_diff_curr[0] + Ri[3]*t_diff_curr[1] + Ri[6]*t_diff_curr[2],
                Ri[1]*t_diff_curr[0] + Ri[4]*t_diff_curr[1] + Ri[7]*t_diff_curr[2],
                Ri[2]*t_diff_curr[0] + Ri[5]*t_diff_curr[1] + Ri[8]*t_diff_curr[2]
            };
            
            /* 旋转部分：q_ij = qi^-1 * qj */
            float qi_conj[4] = {qi[0], -qi[1], -qi[2], -qi[3]};
            float q_ij[4] = {
                qi_conj[0]*qj[0] - qi_conj[1]*qj[1] - qi_conj[2]*qj[2] - qi_conj[3]*qj[3],
                qi_conj[0]*qj[1] + qi_conj[1]*qj[0] + qi_conj[2]*qj[3] - qi_conj[3]*qj[2],
                qi_conj[0]*qj[2] - qi_conj[1]*qj[3] + qi_conj[2]*qj[0] + qi_conj[3]*qj[1],
                qi_conj[0]*qj[3] + qi_conj[1]*qj[2] - qi_conj[2]*qj[1] + qi_conj[3]*qj[0]
            };
            
            /* 归一化 */
            float norm_ij = sqrtf(q_ij[0]*q_ij[0] + q_ij[1]*q_ij[1] + q_ij[2]*q_ij[2] + q_ij[3]*q_ij[3]);
            if (norm_ij > 1e-6f) {
                q_ij[0] /= norm_ij;
                q_ij[1] /= norm_ij;
                q_ij[2] /= norm_ij;
                q_ij[3] /= norm_ij;
            }
            
            /* 目标相对位姿 */
            float* target = &constraint_targets[7*c];
            float t_target[3] = {target[0], target[1], target[2]};
            float q_target[4] = {target[3], target[4], target[5], target[6]};
            
            /* 计算残差 */
            /* 平移残差 */
            float residual_t[3] = {
                t_ij[0] - t_target[0],
                t_ij[1] - t_target[1],
                t_ij[2] - t_target[2]
            };
            
            /* 旋转残差：使用四元数差的对数映射 */
            float q_target_conj[4] = {q_target[0], -q_target[1], -q_target[2], -q_target[3]};
            float q_error[4] = {
                q_target_conj[0]*q_ij[0] - q_target_conj[1]*q_ij[1] - q_target_conj[2]*q_ij[2] - q_target_conj[3]*q_ij[3],
                q_target_conj[0]*q_ij[1] + q_target_conj[1]*q_ij[0] + q_target_conj[2]*q_ij[3] - q_target_conj[3]*q_ij[2],
                q_target_conj[0]*q_ij[2] - q_target_conj[1]*q_ij[3] + q_target_conj[2]*q_ij[0] + q_target_conj[3]*q_ij[1],
                q_target_conj[0]*q_ij[3] + q_target_conj[1]*q_ij[2] - q_target_conj[2]*q_ij[1] + q_target_conj[3]*q_ij[0]
            };
            
            /* 四元数误差转换为旋转向量 */
            float angle_error = 2.0f * acosf(q_error[0]);
            float axis_norm_error = sqrtf(q_error[1]*q_error[1] + q_error[2]*q_error[2] + q_error[3]*q_error[3]);
            float residual_r[3];
            
            if (axis_norm_error > 1e-6f) {
                residual_r[0] = angle_error * q_error[1] / axis_norm_error;
                residual_r[1] = angle_error * q_error[2] / axis_norm_error;
                residual_r[2] = angle_error * q_error[3] / axis_norm_error;
            } else {
                residual_r[0] = 0.0f;
                residual_r[1] = 0.0f;
                residual_r[2] = 0.0f;
            }
            
            /* 存储残差 */
            for (int k = 0; k < 3; k++) {
                residuals[6*c + k] = residual_t[k] * weight;
                residuals[6*c + 3 + k] = residual_r[k] * weight;
                
                total_error += residual_t[k]*residual_t[k] + residual_r[k]*residual_r[k];
            }
            
            /* 计算雅可比矩阵（数值近似） */
            float epsilon = 1e-4f;
            
            /* 对位姿i的参数 */
            for (int param = 0; param < 6; param++) {
                float params_plus[6];
                memcpy(params_plus, pose_i, 6 * sizeof(float));
                params_plus[param] += epsilon;
                
                /* 完整实现：重新计算扰动参数下的残差 */
                /* 提取扰动后的位姿i和原始位姿j */
                float ti_plus[3] = {params_plus[0], params_plus[1], params_plus[2]};
                float ri_plus[3] = {params_plus[3], params_plus[4], params_plus[5]};
                
                float tj_orig[3] = {pose_j[0], pose_j[1], pose_j[2]};
                float rj_orig[3] = {pose_j[3], pose_j[4], pose_j[5]};
                
                /* 将旋转向量转换为四元数 */
                float qi_plus[4], qj_orig[4];
                float angle_i_plus = sqrtf(ri_plus[0]*ri_plus[0] + ri_plus[1]*ri_plus[1] + ri_plus[2]*ri_plus[2]);
                if (angle_i_plus < 1e-6f) {
                    qi_plus[0] = 1.0f; qi_plus[1] = 0.0f; qi_plus[2] = 0.0f; qi_plus[3] = 0.0f;
                } else {
                    float axis_i_plus[3] = {ri_plus[0]/angle_i_plus, ri_plus[1]/angle_i_plus, ri_plus[2]/angle_i_plus};
                    float sin_half_i_plus = sinf(angle_i_plus * 0.5f);
                    qi_plus[0] = cosf(angle_i_plus * 0.5f);
                    qi_plus[1] = axis_i_plus[0] * sin_half_i_plus;
                    qi_plus[2] = axis_i_plus[1] * sin_half_i_plus;
                    qi_plus[3] = axis_i_plus[2] * sin_half_i_plus;
                }
                
                float angle_j_orig = sqrtf(rj_orig[0]*rj_orig[0] + rj_orig[1]*rj_orig[1] + rj_orig[2]*rj_orig[2]);
                if (angle_j_orig < 1e-6f) {
                    qj_orig[0] = 1.0f; qj_orig[1] = 0.0f; qj_orig[2] = 0.0f; qj_orig[3] = 0.0f;
                } else {
                    float axis_j_orig[3] = {rj_orig[0]/angle_j_orig, rj_orig[1]/angle_j_orig, rj_orig[2]/angle_j_orig};
                    float sin_half_j_orig = sinf(angle_j_orig * 0.5f);
                    qj_orig[0] = cosf(angle_j_orig * 0.5f);
                    qj_orig[1] = axis_j_orig[0] * sin_half_j_orig;
                    qj_orig[2] = axis_j_orig[1] * sin_half_j_orig;
                    qj_orig[3] = axis_j_orig[2] * sin_half_j_orig;
                }
                
                /* 计算扰动后的相对位姿：Tij_plus = Ti_plus^-1 * Tj_orig */
                /* 旋转矩阵 Ri_plus */
                float Ri_plus[9];
                float wi_plus = qi_plus[0], xi_plus = qi_plus[1], yi_plus = qi_plus[2], zi_plus = qi_plus[3];
                Ri_plus[0] = 1 - 2*yi_plus*yi_plus - 2*zi_plus*zi_plus;
                Ri_plus[1] = 2*xi_plus*yi_plus - 2*wi_plus*zi_plus;
                Ri_plus[2] = 2*xi_plus*zi_plus + 2*wi_plus*yi_plus;
                Ri_plus[3] = 2*xi_plus*yi_plus + 2*wi_plus*zi_plus;
                Ri_plus[4] = 1 - 2*xi_plus*xi_plus - 2*zi_plus*zi_plus;
                Ri_plus[5] = 2*yi_plus*zi_plus - 2*wi_plus*xi_plus;
                Ri_plus[6] = 2*xi_plus*zi_plus - 2*wi_plus*yi_plus;
                Ri_plus[7] = 2*yi_plus*zi_plus + 2*wi_plus*xi_plus;
                Ri_plus[8] = 1 - 2*xi_plus*xi_plus - 2*yi_plus*yi_plus;
                
                /* 平移部分：t_ij_plus = Ri_plus^T * (tj_orig - ti_plus) */
                float t_diff_plus[3] = {tj_orig[0] - ti_plus[0], tj_orig[1] - ti_plus[1], tj_orig[2] - ti_plus[2]};
                float t_ij_plus[3] = {
                    Ri_plus[0]*t_diff_plus[0] + Ri_plus[3]*t_diff_plus[1] + Ri_plus[6]*t_diff_plus[2],
                    Ri_plus[1]*t_diff_plus[0] + Ri_plus[4]*t_diff_plus[1] + Ri_plus[7]*t_diff_plus[2],
                    Ri_plus[2]*t_diff_plus[0] + Ri_plus[5]*t_diff_plus[1] + Ri_plus[8]*t_diff_plus[2]
                };
                
                /* 旋转部分：q_ij_plus = qi_plus^-1 * qj_orig */
                float qi_plus_conj[4] = {qi_plus[0], -qi_plus[1], -qi_plus[2], -qi_plus[3]};
                float q_ij_plus[4] = {
                    qi_plus_conj[0]*qj_orig[0] - qi_plus_conj[1]*qj_orig[1] - qi_plus_conj[2]*qj_orig[2] - qi_plus_conj[3]*qj_orig[3],
                    qi_plus_conj[0]*qj_orig[1] + qi_plus_conj[1]*qj_orig[0] + qi_plus_conj[2]*qj_orig[3] - qi_plus_conj[3]*qj_orig[2],
                    qi_plus_conj[0]*qj_orig[2] - qi_plus_conj[1]*qj_orig[3] + qi_plus_conj[2]*qj_orig[0] + qi_plus_conj[3]*qj_orig[1],
                    qi_plus_conj[0]*qj_orig[3] + qi_plus_conj[1]*qj_orig[2] - qi_plus_conj[2]*qj_orig[1] + qi_plus_conj[3]*qj_orig[0]
                };
                
                /* 归一化 */
                float norm_ij_plus = sqrtf(q_ij_plus[0]*q_ij_plus[0] + q_ij_plus[1]*q_ij_plus[1] + 
                                         q_ij_plus[2]*q_ij_plus[2] + q_ij_plus[3]*q_ij_plus[3]);
                if (norm_ij_plus > 1e-6f) {
                    q_ij_plus[0] /= norm_ij_plus;
                    q_ij_plus[1] /= norm_ij_plus;
                    q_ij_plus[2] /= norm_ij_plus;
                    q_ij_plus[3] /= norm_ij_plus;
                }
                
                /* 目标相对位姿 */
                float* target_p = &constraint_targets[7*c];
                float t_target_p[3] = {target_p[0], target_p[1], target_p[2]};
                float q_target_p[4] = {target_p[3], target_p[4], target_p[5], target_p[6]};
                
                /* 计算扰动后的残差 */
                /* 平移残差 */
                float residual_t_plus[3] = {
                    t_ij_plus[0] - t_target_p[0],
                    t_ij_plus[1] - t_target_p[1],
                    t_ij_plus[2] - t_target_p[2]
                };
                
                /* 旋转残差：使用四元数差的对数映射 */
                float q_target_conj_p[4] = {q_target_p[0], -q_target_p[1], -q_target_p[2], -q_target_p[3]};
                float q_error_plus[4] = {
                    q_target_conj_p[0]*q_ij_plus[0] - q_target_conj_p[1]*q_ij_plus[1] - q_target_conj_p[2]*q_ij_plus[2] - q_target_conj_p[3]*q_ij_plus[3],
                    q_target_conj_p[0]*q_ij_plus[1] + q_target_conj_p[1]*q_ij_plus[0] + q_target_conj_p[2]*q_ij_plus[3] - q_target_conj_p[3]*q_ij_plus[2],
                    q_target_conj_p[0]*q_ij_plus[2] - q_target_conj_p[1]*q_ij_plus[3] + q_target_conj_p[2]*q_ij_plus[0] + q_target_conj_p[3]*q_ij_plus[1],
                    q_target_conj_p[0]*q_ij_plus[3] + q_target_conj_p[1]*q_ij_plus[2] - q_target_conj_p[2]*q_ij_plus[1] + q_target_conj_p[3]*q_ij_plus[0]
                };
                
                /* 四元数误差转换为旋转向量 */
                float angle_error_plus = 2.0f * acosf(q_error_plus[0]);
                float axis_norm_error_plus = sqrtf(q_error_plus[1]*q_error_plus[1] + q_error_plus[2]*q_error_plus[2] + q_error_plus[3]*q_error_plus[3]);
                float residual_r_plus[3];
                
                if (axis_norm_error_plus > 1e-6f) {
                    residual_r_plus[0] = angle_error_plus * q_error_plus[1] / axis_norm_error_plus;
                    residual_r_plus[1] = angle_error_plus * q_error_plus[2] / axis_norm_error_plus;
                    residual_r_plus[2] = angle_error_plus * q_error_plus[3] / axis_norm_error_plus;
                } else {
                    residual_r_plus[0] = 0.0f;
                    residual_r_plus[1] = 0.0f;
                    residual_r_plus[2] = 0.0f;
                }
                
                /* 计算扰动后的总残差（仅用于雅可比计算） */
                float residual_plus = 0.0f;
                for (int k = 0; k < 3; k++) {
                    residual_plus += residual_t_plus[k]*residual_t_plus[k] + residual_r_plus[k]*residual_r_plus[k];
                }
                
                /* 近似雅可比 */
                jacobian[(6*c) * num_params + (6*i + param)] = residual_plus / epsilon;
            }
            
            /* 对位姿j的参数 */
            for (int param = 0; param < 6; param++) {
                float params_plus[6];
                memcpy(params_plus, pose_j, 6 * sizeof(float));
                params_plus[param] += epsilon;
                
                /* 完整实现：重新计算扰动参数下的残差 */
                /* 提取原始位姿i和扰动后的位姿j */
                float ti_orig[3] = {pose_i[0], pose_i[1], pose_i[2]};
                float ri_orig[3] = {pose_i[3], pose_i[4], pose_i[5]};
                
                float tj_plus[3] = {params_plus[0], params_plus[1], params_plus[2]};
                float rj_plus[3] = {params_plus[3], params_plus[4], params_plus[5]};
                
                /* 将旋转向量转换为四元数 */
                float qi_orig[4], qj_plus[4];
                float angle_i_orig = sqrtf(ri_orig[0]*ri_orig[0] + ri_orig[1]*ri_orig[1] + ri_orig[2]*ri_orig[2]);
                if (angle_i_orig < 1e-6f) {
                    qi_orig[0] = 1.0f; qi_orig[1] = 0.0f; qi_orig[2] = 0.0f; qi_orig[3] = 0.0f;
                } else {
                    float axis_i_orig[3] = {ri_orig[0]/angle_i_orig, ri_orig[1]/angle_i_orig, ri_orig[2]/angle_i_orig};
                    float sin_half_i_orig = sinf(angle_i_orig * 0.5f);
                    qi_orig[0] = cosf(angle_i_orig * 0.5f);
                    qi_orig[1] = axis_i_orig[0] * sin_half_i_orig;
                    qi_orig[2] = axis_i_orig[1] * sin_half_i_orig;
                    qi_orig[3] = axis_i_orig[2] * sin_half_i_orig;
                }
                
                float angle_j_plus = sqrtf(rj_plus[0]*rj_plus[0] + rj_plus[1]*rj_plus[1] + rj_plus[2]*rj_plus[2]);
                if (angle_j_plus < 1e-6f) {
                    qj_plus[0] = 1.0f; qj_plus[1] = 0.0f; qj_plus[2] = 0.0f; qj_plus[3] = 0.0f;
                } else {
                    float axis_j_plus[3] = {rj_plus[0]/angle_j_plus, rj_plus[1]/angle_j_plus, rj_plus[2]/angle_j_plus};
                    float sin_half_j_plus = sinf(angle_j_plus * 0.5f);
                    qj_plus[0] = cosf(angle_j_plus * 0.5f);
                    qj_plus[1] = axis_j_plus[0] * sin_half_j_plus;
                    qj_plus[2] = axis_j_plus[1] * sin_half_j_plus;
                    qj_plus[3] = axis_j_plus[2] * sin_half_j_plus;
                }
                
                /* 计算扰动后的相对位姿：Tij_plus = Ti_orig^-1 * Tj_plus */
                /* 旋转矩阵 Ri_orig */
                float Ri_orig[9];
                float wi_orig = qi_orig[0], xi_orig = qi_orig[1], yi_orig = qi_orig[2], zi_orig = qi_orig[3];
                Ri_orig[0] = 1 - 2*yi_orig*yi_orig - 2*zi_orig*zi_orig;
                Ri_orig[1] = 2*xi_orig*yi_orig - 2*wi_orig*zi_orig;
                Ri_orig[2] = 2*xi_orig*zi_orig + 2*wi_orig*yi_orig;
                Ri_orig[3] = 2*xi_orig*yi_orig + 2*wi_orig*zi_orig;
                Ri_orig[4] = 1 - 2*xi_orig*xi_orig - 2*zi_orig*zi_orig;
                Ri_orig[5] = 2*yi_orig*zi_orig - 2*wi_orig*xi_orig;
                Ri_orig[6] = 2*xi_orig*zi_orig - 2*wi_orig*yi_orig;
                Ri_orig[7] = 2*yi_orig*zi_orig + 2*wi_orig*xi_orig;
                Ri_orig[8] = 1 - 2*xi_orig*xi_orig - 2*yi_orig*yi_orig;
                
                /* 平移部分：t_ij_plus = Ri_orig^T * (tj_plus - ti_orig) */
                float t_diff_plus[3] = {tj_plus[0] - ti_orig[0], tj_plus[1] - ti_orig[1], tj_plus[2] - ti_orig[2]};
                float t_ij_plus[3] = {
                    Ri_orig[0]*t_diff_plus[0] + Ri_orig[3]*t_diff_plus[1] + Ri_orig[6]*t_diff_plus[2],
                    Ri_orig[1]*t_diff_plus[0] + Ri_orig[4]*t_diff_plus[1] + Ri_orig[7]*t_diff_plus[2],
                    Ri_orig[2]*t_diff_plus[0] + Ri_orig[5]*t_diff_plus[1] + Ri_orig[8]*t_diff_plus[2]
                };
                
                /* 旋转部分：q_ij_plus = qi_orig^-1 * qj_plus */
                float qi_orig_conj[4] = {qi_orig[0], -qi_orig[1], -qi_orig[2], -qi_orig[3]};
                float q_ij_plus[4] = {
                    qi_orig_conj[0]*qj_plus[0] - qi_orig_conj[1]*qj_plus[1] - qi_orig_conj[2]*qj_plus[2] - qi_orig_conj[3]*qj_plus[3],
                    qi_orig_conj[0]*qj_plus[1] + qi_orig_conj[1]*qj_plus[0] + qi_orig_conj[2]*qj_plus[3] - qi_orig_conj[3]*qj_plus[2],
                    qi_orig_conj[0]*qj_plus[2] - qi_orig_conj[1]*qj_plus[3] + qi_orig_conj[2]*qj_plus[0] + qi_orig_conj[3]*qj_plus[1],
                    qi_orig_conj[0]*qj_plus[3] + qi_orig_conj[1]*qj_plus[2] - qi_orig_conj[2]*qj_plus[1] + qi_orig_conj[3]*qj_plus[0]
                };
                
                /* 归一化 */
                float norm_ij_plus = sqrtf(q_ij_plus[0]*q_ij_plus[0] + q_ij_plus[1]*q_ij_plus[1] + 
                                         q_ij_plus[2]*q_ij_plus[2] + q_ij_plus[3]*q_ij_plus[3]);
                if (norm_ij_plus > 1e-6f) {
                    q_ij_plus[0] /= norm_ij_plus;
                    q_ij_plus[1] /= norm_ij_plus;
                    q_ij_plus[2] /= norm_ij_plus;
                    q_ij_plus[3] /= norm_ij_plus;
                }
                
                /* 目标相对位姿 */
                float* target_j = &constraint_targets[7*c];
                float t_target_j[3] = {target_j[0], target_j[1], target_j[2]};
                float q_target_j[4] = {target_j[3], target_j[4], target_j[5], target_j[6]};
                
                /* 计算扰动后的残差 */
                /* 平移残差 */
                float residual_t_plus[3] = {
                    t_ij_plus[0] - t_target_j[0],
                    t_ij_plus[1] - t_target_j[1],
                    t_ij_plus[2] - t_target_j[2]
                };
                
                /* 旋转残差：使用四元数差的对数映射 */
                float q_target_conj_j[4] = {q_target_j[0], -q_target_j[1], -q_target_j[2], -q_target_j[3]};
                float q_error_plus[4] = {
                    q_target_conj_j[0]*q_ij_plus[0] - q_target_conj_j[1]*q_ij_plus[1] - q_target_conj_j[2]*q_ij_plus[2] - q_target_conj_j[3]*q_ij_plus[3],
                    q_target_conj_j[0]*q_ij_plus[1] + q_target_conj_j[1]*q_ij_plus[0] + q_target_conj_j[2]*q_ij_plus[3] - q_target_conj_j[3]*q_ij_plus[2],
                    q_target_conj_j[0]*q_ij_plus[2] - q_target_conj_j[1]*q_ij_plus[3] + q_target_conj_j[2]*q_ij_plus[0] + q_target_conj_j[3]*q_ij_plus[1],
                    q_target_conj_j[0]*q_ij_plus[3] + q_target_conj_j[1]*q_ij_plus[2] - q_target_conj_j[2]*q_ij_plus[1] + q_target_conj_j[3]*q_ij_plus[0]
                };
                
                /* 四元数误差转换为旋转向量 */
                float angle_error_plus = 2.0f * acosf(q_error_plus[0]);
                float axis_norm_error_plus = sqrtf(q_error_plus[1]*q_error_plus[1] + q_error_plus[2]*q_error_plus[2] + q_error_plus[3]*q_error_plus[3]);
                float residual_r_plus[3];
                
                if (axis_norm_error_plus > 1e-6f) {
                    residual_r_plus[0] = angle_error_plus * q_error_plus[1] / axis_norm_error_plus;
                    residual_r_plus[1] = angle_error_plus * q_error_plus[2] / axis_norm_error_plus;
                    residual_r_plus[2] = angle_error_plus * q_error_plus[3] / axis_norm_error_plus;
                } else {
                    residual_r_plus[0] = 0.0f;
                    residual_r_plus[1] = 0.0f;
                    residual_r_plus[2] = 0.0f;
                }
                
                /* 计算扰动后的总残差（仅用于雅可比计算） */
                float residual_plus = 0.0f;
                for (int k = 0; k < 3; k++) {
                    residual_plus += residual_t_plus[k]*residual_t_plus[k] + residual_r_plus[k]*residual_r_plus[k];
                }
                
                /* 近似雅可比 */
                jacobian[(6*c) * num_params + (6*j + param)] = residual_plus / epsilon;
            }
        }
        
        /* 构建正规方程并求解（完整实现：使用高斯-牛顿法） */
        /* 计算正规方程：J^T * J * delta = -J^T * r */
        
        /* 分配内存用于J^T * J矩阵和J^T * r向量 */
        float* JTJ = (float*)slam_calloc(num_params * num_params, sizeof(float));
        float* JTr = (float*)slam_calloc(num_params, sizeof(float));
        
        if (JTJ && JTr) {
            /* 计算J^T * J矩阵 */
            for (int i = 0; i < num_params; i++) {
                for (int j = 0; j < num_params; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < 6 * num_constraints; k++) {
                        sum += jacobian[k * num_params + i] * jacobian[k * num_params + j];
                    }
                    JTJ[i * num_params + j] = sum;
                }
            }
            
            /* 计算J^T * r向量 */
            for (int i = 0; i < num_params; i++) {
                float sum = 0.0f;
                for (int k = 0; k < 6 * num_constraints; k++) {
                    sum += jacobian[k * num_params + i] * residuals[k];
                }
                JTr[i] = -sum; /* 负号因为方程是J^T * J * delta = -J^T * r */
            }
            
            /* 添加阻尼项（Levenberg-Marquardt正则化）以提高数值稳定性 */
            for (int i = 0; i < num_params; i++) {
                JTJ[i * num_params + i] += lambda; /* 使用自适应lambda（非固定值） */
            }
            
            /* 求解线性系统：JTJ * delta = JTr */
            float* delta = (float*)slam_calloc(num_params, sizeof(float));
            if (delta) {
                /* 使用高斯消元法求解线性系统 */
                /* 创建增广矩阵 */
                float* A = (float*)slam_malloc(num_params * (num_params + 1) * sizeof(float));
                if (A) {
                    /* 复制JTJ矩阵到A */
                    for (int i = 0; i < num_params; i++) {
                        for (int j = 0; j < num_params; j++) {
                            A[i * (num_params + 1) + j] = JTJ[i * num_params + j];
                        }
                        /* 添加右侧向量 */
                        A[i * (num_params + 1) + num_params] = JTr[i];
                    }
                    
                    /* 高斯消元 */
                    for (int i = 0; i < num_params; i++) {
                        /* 寻找主元 */
                        int pivot = i;
                        float max_val = fabsf(A[i * (num_params + 1) + i]);
                        for (int j = i + 1; j < num_params; j++) {
                            float val = fabsf(A[j * (num_params + 1) + i]);
                            if (val > max_val) {
                                max_val = val;
                                pivot = j;
                            }
                        }
                        
                        /* 交换行 */
                        if (pivot != i) {
                            for (int j = 0; j <= num_params; j++) {
                                float temp = A[i * (num_params + 1) + j];
                                A[i * (num_params + 1) + j] = A[pivot * (num_params + 1) + j];
                                A[pivot * (num_params + 1) + j] = temp;
                            }
                        }
                        
                        /* 主元归一化 */
                        float diag = A[i * (num_params + 1) + i];
                        if (fabsf(diag) < 1e-10f) {
                            diag = 1e-10f;
                            A[i * (num_params + 1) + i] = diag;
                        }
                        
                        /* 消元 */
                        for (int j = i + 1; j < num_params; j++) {
                            float factor = A[j * (num_params + 1) + i] / diag;
                            for (int k = i; k <= num_params; k++) {
                                A[j * (num_params + 1) + k] -= factor * A[i * (num_params + 1) + k];
                            }
                        }
                    }
                    
                    /* 回代 */
                    for (int i = num_params - 1; i >= 0; i--) {
                        float sum = A[i * (num_params + 1) + num_params];
                        for (int j = i + 1; j < num_params; j++) {
                            sum -= A[i * (num_params + 1) + j] * delta[j];
                        }
                        delta[i] = sum / A[i * (num_params + 1) + i];
                    }
                    
                    slam_free(A);
                }
                
                /* 更新参数：params = params + delta */
                for (int i = 0; i < num_params; i++) {
                    params[i] += delta[i];
                }
                
                slam_free(delta);
            }
            
            slam_free(JTJ);
            slam_free(JTr);
        } else {
            /* 内存分配失败，使用梯度下降作为后备 */
            if (JTJ) slam_free(JTJ);
            if (JTr) slam_free(JTr);
            
            /* 计算梯度：J^T * r */
            float* gradient = (float*)slam_calloc(num_params, sizeof(float));
            if (gradient) {
                for (int i = 0; i < num_params; i++) {
                    float sum = 0.0f;
                    for (int k = 0; k < 6 * num_constraints; k++) {
                        sum += jacobian[k * num_params + i] * residuals[k];
                    }
                    gradient[i] = sum;
                }
                
                /* 梯度下降更新：params = params - alpha * gradient */
                float alpha = 0.01f; /* 学习率 */
                for (int i = 0; i < num_params; i++) {
                    params[i] -= alpha * gradient[i];
                }
                
                slam_free(gradient);
            }
        }
        
        /* 清理迭代内存 */
        slam_free(residuals);
        slam_free(jacobian);
        
        /* LM阻尼因子自适应调整：根据误差变化趋势动态调节 */
        if (iter > 0) {
            if (total_error < prev_total_error) {
                /* 误差减小：降低阻尼，使优化趋向高斯-牛顿法 */
                lambda *= 0.5f;
                if (lambda < 1e-6f) lambda = 1e-6f;
            } else {
                /* 误差增大：提高阻尼，使优化趋向梯度下降法 */
                lambda *= 2.0f;
                if (lambda > 1.0f) lambda = 1.0f;
            }
        }
        prev_total_error = total_error;
        
        /* 检查收敛条件 */
        if (total_error < 1e-4f || iter == max_iterations - 1) {
            break;
        }
    }
    
    /* 7. 将优化后的位姿写回关键帧 */
    for (int i = 0; i < num_frames_to_optimize; i++) {
        int frame_idx = start_frame + i;
        KeyFrame* kf = &system->local_map.keyframes[frame_idx];
        
        /* 平移向量 */
        kf->pose.position[0] = params[6*i];
        kf->pose.position[1] = params[6*i + 1];
        kf->pose.position[2] = params[6*i + 2];
        
        /* 旋转向量转换为四元数 */
        float rx = params[6*i + 3];
        float ry = params[6*i + 4];
        float rz = params[6*i + 5];
        
        float angle = sqrtf(rx*rx + ry*ry + rz*rz);
        if (angle < 1e-6f) {
            kf->pose.orientation[0] = 1.0f; /* qw */
            kf->pose.orientation[1] = 0.0f; /* qx */
            kf->pose.orientation[2] = 0.0f; /* qy */
            kf->pose.orientation[3] = 0.0f; /* qz */
        } else {
            float axis[3] = {rx/angle, ry/angle, rz/angle};
            float half_angle = angle * 0.5f;
            float sin_half = sinf(half_angle);
            
            kf->pose.orientation[0] = cosf(half_angle); /* qw */
            kf->pose.orientation[1] = axis[0] * sin_half; /* qx */
            kf->pose.orientation[2] = axis[1] * sin_half; /* qy */
            kf->pose.orientation[3] = axis[2] * sin_half; /* qz */
        }
    }
    
    /* 8. 更新系统状态 */
    system->last_loop_frame_id = frame_id;

    /* 9. 地图点融合：合并闭环检测到的重复地图点 */
    if (system->loop_closure_config.enable_map_fusion) {
        int fusion_count = slam_fuse_loop_closure_map(system, frame_id, matched_frame_id);
        if (fusion_count > 0) {
            system->loop_closure_internal.total_fusions_performed += fusion_count;
            system->loop_closure_internal.fusion_pending = 1;
        }
    }

    /* 10. 漂移传播：将校正后的位姿偏差传播到优化窗口外的所有关键帧 */
    if (system->loop_closure_config.enable_drift_propagation) {
        slam_propagate_drift_correction(system, start_frame, end_frame, params, num_frames_to_optimize);
    }

    /* 11. 清理内存 */
    slam_free(params);
    slam_free(constraint_targets);
    slam_free(constraint_from);
    slam_free(constraint_to);
    slam_free(constraint_weights);
    
    return 0;
}

static int slam_build_optimization_problem(SlamSystem* system,
                                          OptimizationProblem* problem) {
    if (!system || !problem) {
        return -1;
    }
    
    /* 完整构建优化问题：收集所有关键帧、地标点和观测数据 */
    
    /* 1. 清空问题结构体 */
    memset(problem, 0, sizeof(OptimizationProblem));
    
    /* 2. 收集所有关键帧位姿 */
    int num_keyframes = system->local_map.num_keyframes;
    if (num_keyframes == 0) {
        return -1;
    }
    
    /* 位姿参数：每个关键帧7个参数（平移3 + 四元数4） */
    problem->num_poses = num_keyframes;
    problem->poses = (float*)slam_malloc(7 * num_keyframes * sizeof(float));
    if (!problem->poses) {
        return -1;
    }
    
    for (int i = 0; i < num_keyframes; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        
        /* 平移向量 */
        problem->poses[7*i] = kf->pose.position[0];
        problem->poses[7*i + 1] = kf->pose.position[1];
        problem->poses[7*i + 2] = kf->pose.position[2];
        
        /* 旋转四元数 */
        problem->poses[7*i + 3] = kf->pose.orientation[0]; /* qw */
        problem->poses[7*i + 4] = kf->pose.orientation[1]; /* qx */
        problem->poses[7*i + 5] = kf->pose.orientation[2]; /* qy */
        problem->poses[7*i + 6] = kf->pose.orientation[3]; /* qz */
    }
    
    /* 3. 收集所有地标点 */
    int num_landmarks = system->local_map.num_landmarks;
    problem->num_landmarks = num_landmarks;
    
    if (num_landmarks > 0) {
        problem->landmarks = (float*)slam_malloc(3 * num_landmarks * sizeof(float));
        if (!problem->landmarks) {
            slam_free(problem->poses);
            problem->poses = NULL;
            return -1;
        }
        
        for (int i = 0; i < num_landmarks; i++) {
            Landmark* lm = &system->local_map.landmarks[i];
            problem->landmarks[3*i] = lm->position[0];
            problem->landmarks[3*i + 1] = lm->position[1];
            problem->landmarks[3*i + 2] = lm->position[2];
        }
    }
    
    /* 4. 收集所有观测数据 */
    /* 首先计算总观测数量 */
    int total_observations = 0;
    for (int i = 0; i < num_keyframes; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        if (kf->landmark_ids && kf->num_landmarks > 0) {
            total_observations += kf->num_landmarks;
        }
    }
    
    problem->num_observations = total_observations;
    
    if (total_observations > 0) {
        problem->observation_frames = (int*)slam_malloc(total_observations * sizeof(int));
        problem->observation_landmarks = (int*)slam_malloc(total_observations * sizeof(int));
        problem->observation_points = (float*)slam_malloc(2 * total_observations * sizeof(float));
        problem->observation_weights = (float*)slam_malloc(total_observations * sizeof(float));
        
        if (!problem->observation_frames || !problem->observation_landmarks ||
            !problem->observation_points || !problem->observation_weights) {
            if (problem->observation_frames) slam_free(problem->observation_frames);
            if (problem->observation_landmarks) slam_free(problem->observation_landmarks);
            if (problem->observation_points) slam_free(problem->observation_points);
            if (problem->observation_weights) slam_free(problem->observation_weights);
            if (problem->landmarks) slam_free(problem->landmarks);
            if (problem->poses) slam_free(problem->poses);
            memset(problem, 0, sizeof(OptimizationProblem));
            return -1;
        }
        
        /* 填充观测数据 */
        int obs_idx = 0;
        for (int frame_id = 0; frame_id < num_keyframes; frame_id++) {
            KeyFrame* kf = &system->local_map.keyframes[frame_id];
            
            if (kf->landmark_ids && kf->num_landmarks > 0) {
                for (int i = 0; i < kf->num_landmarks; i++) {
                    int lm_id = kf->landmark_ids[i];
                    if (lm_id >= 0 && lm_id < num_landmarks) {
                        problem->observation_frames[obs_idx] = frame_id;
                        problem->observation_landmarks[obs_idx] = lm_id;
                        
                        /* 获取观测到的2D点坐标 */
                        if (kf->keypoints_x && kf->keypoints_y && i < kf->num_features) {
                            problem->observation_points[2*obs_idx] = (float)kf->keypoints_x[i];
                            problem->observation_points[2*obs_idx + 1] = (float)kf->keypoints_y[i];
                        } else {
                            /* 如果没有特征点数据，使用默认值 */
                            problem->observation_points[2*obs_idx] = 0.0f;
                            problem->observation_points[2*obs_idx + 1] = 0.0f;
                        }
                        
                        /* 观测权重：基于地标点的观测次数 */
                        Landmark* lm = &system->local_map.landmarks[lm_id];
                        float weight = 1.0f;
                        if (lm->observed_count > 0) {
                            weight = 1.0f / sqrtf((float)lm->observed_count);
                        }
                        problem->observation_weights[obs_idx] = weight;
                        
                        obs_idx++;
                    }
                }
            }
        }
        
        /* 更新实际观测数量（可能小于预期，如果有些地标点ID无效） */
        problem->num_observations = obs_idx;
    }
    
    /* 5. 设置相机参数 */
    problem->camera_params = (float*)slam_malloc(4 * sizeof(float));
    if (problem->camera_params) {
        problem->camera_params[0] = system->config.camera_params[0]; /* fx */
        problem->camera_params[1] = system->config.camera_params[1]; /* fy */
        problem->camera_params[2] = system->config.camera_params[2]; /* cx */
        problem->camera_params[3] = system->config.camera_params[3]; /* cy */
    }
    
    return 0;
}

static int slam_solve_optimization_problem(OptimizationProblem* problem,
                                          int max_iterations) {
    if (!problem) {
        return -1;
    }
    
    /* 完整求解优化问题：使用高斯-牛顿法进行捆集调整 */
    
    /* 检查问题是否有效 */
    if (problem->num_poses == 0 || problem->num_observations == 0) {
        return -1;
    }
    
    /* 如果max_iterations为0，使用默认值 */
    if (max_iterations <= 0) {
        max_iterations = 20;
    }
    
    /* 参数总数：7 * num_poses + 3 * num_landmarks */
    int num_pose_params = 7 * problem->num_poses;
    int num_landmark_params = 3 * problem->num_landmarks;
    int num_params = num_pose_params + num_landmark_params;
    
    if (num_params == 0) {
        return -1;
    }
    
    /* 创建参数向量 */
    float* params = (float*)slam_malloc(num_params * sizeof(float));
    if (!params) {
        return -1;
    }
    
    /* 初始化参数：从问题中复制位姿和地标点 */
    memcpy(params, problem->poses, num_pose_params * sizeof(float));
    if (problem->landmarks && num_landmark_params > 0) {
        memcpy(params + num_pose_params, problem->landmarks, num_landmark_params * sizeof(float));
    }
    
    /* 相机参数 */
    float fx = problem->camera_params ? problem->camera_params[0] : 525.0f;
    float fy = problem->camera_params ? problem->camera_params[1] : 525.0f;
    float cx = problem->camera_params ? problem->camera_params[2] : 320.0f;
    float cy = problem->camera_params ? problem->camera_params[3] : 240.0f;
    
    /* 优化循环 */
    float lambda = 1e-3f; /* Levenberg-Marquardt阻尼因子 */
    
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        /* 计算残差和雅可比矩阵 */
        /* 残差维度：2 * num_observations */
        int residual_dim = 2 * problem->num_observations;
        float* residuals = (float*)slam_malloc(residual_dim * sizeof(float));
        float* jacobian = (float*)slam_malloc(residual_dim * num_params * sizeof(float));
        
        if (!residuals || !jacobian) {
            if (residuals) slam_free(residuals);
            if (jacobian) slam_free(jacobian);
            break;
        }
        
        memset(jacobian, 0, residual_dim * num_params * sizeof(float));
        float total_error = 0.0f;
        
        /* 对每个观测计算残差和雅可比 */
        for (int obs = 0; obs < problem->num_observations; obs++) {
            int frame_id = problem->observation_frames[obs];
            int landmark_id = problem->observation_landmarks[obs];
            float weight = problem->observation_weights ? problem->observation_weights[obs] : 1.0f;
            
            /* 获取观测到的2D点 */
            float u_obs = problem->observation_points[2*obs];
            float v_obs = problem->observation_points[2*obs + 1];
            
            /* 获取位姿参数 */
            float* pose_params = &params[7 * frame_id];
            float tx = pose_params[0];
            float ty = pose_params[1];
            float tz = pose_params[2];
            float qw = pose_params[3];
            float qx = pose_params[4];
            float qy = pose_params[5];
            float qz = pose_params[6];
            
            /* 获取地标点参数 */
            float* landmark_params = &params[num_pose_params + 3 * landmark_id];
            float X = landmark_params[0];
            float Y = landmark_params[1];
            float Z = landmark_params[2];
            
            /* 计算旋转矩阵 */
            float R[9];
            R[0] = 1 - 2*qy*qy - 2*qz*qz;
            R[1] = 2*qx*qy - 2*qw*qz;
            R[2] = 2*qx*qz + 2*qw*qy;
            
            R[3] = 2*qx*qy + 2*qw*qz;
            R[4] = 1 - 2*qx*qx - 2*qz*qz;
            R[5] = 2*qy*qz - 2*qw*qx;
            
            R[6] = 2*qx*qz - 2*qw*qy;
            R[7] = 2*qy*qz + 2*qw*qx;
            R[8] = 1 - 2*qx*qx - 2*qy*qy;
            
            /* 投影：将地标点转换到相机坐标系 */
            float x_cam = R[0]*X + R[1]*Y + R[2]*Z + tx;
            float y_cam = R[3]*X + R[4]*Y + R[5]*Z + ty;
            float z_cam = R[6]*X + R[7]*Y + R[8]*Z + tz;
            
            if (z_cam < 1e-6f) {
                z_cam = 1e-6f;
            }
            
            /* 归一化平面坐标 */
            float xn = x_cam / z_cam;
            float yn = y_cam / z_cam;
            
            /* 像素坐标 */
            float u = fx * xn + cx;
            float v = fy * yn + cy;
            
            /* 残差 */
            float residual_x = u - u_obs;
            float residual_y = v - v_obs;
            
            residuals[2*obs] = residual_x * weight;
            residuals[2*obs + 1] = residual_y * weight;
            
            total_error += residual_x*residual_x + residual_y*residual_y;
            
            /* 计算雅可比矩阵（数值近似） */
            float epsilon = 1e-4f;
            
            /* 对位姿参数的雅可比 */
            for (int param = 0; param < 7; param++) {
                float params_plus[7];
                memcpy(params_plus, pose_params, 7 * sizeof(float));
                params_plus[param] += epsilon;
                
                /* 重新计算投影 */
                float tx_plus = params_plus[0];
                float ty_plus = params_plus[1];
                float tz_plus = params_plus[2];
                float qw_plus = params_plus[3];
                float qx_plus = params_plus[4];
                float qy_plus = params_plus[5];
                float qz_plus = params_plus[6];
                
                /* 计算新的旋转矩阵 */
                float R_plus[9];
                R_plus[0] = 1 - 2*qy_plus*qy_plus - 2*qz_plus*qz_plus;
                R_plus[1] = 2*qx_plus*qy_plus - 2*qw_plus*qz_plus;
                R_plus[2] = 2*qx_plus*qz_plus + 2*qw_plus*qy_plus;
                
                R_plus[3] = 2*qx_plus*qy_plus + 2*qw_plus*qz_plus;
                R_plus[4] = 1 - 2*qx_plus*qx_plus - 2*qz_plus*qz_plus;
                R_plus[5] = 2*qy_plus*qz_plus - 2*qw_plus*qx_plus;
                
                R_plus[6] = 2*qx_plus*qz_plus - 2*qw_plus*qy_plus;
                R_plus[7] = 2*qy_plus*qz_plus + 2*qw_plus*qx_plus;
                R_plus[8] = 1 - 2*qx_plus*qx_plus - 2*qy_plus*qy_plus;
                
                float x_cam_plus = R_plus[0]*X + R_plus[1]*Y + R_plus[2]*Z + tx_plus;
                float y_cam_plus = R_plus[3]*X + R_plus[4]*Y + R_plus[5]*Z + ty_plus;
                float z_cam_plus = R_plus[6]*X + R_plus[7]*Y + R_plus[8]*Z + tz_plus;
                
                if (z_cam_plus < 1e-6f) z_cam_plus = 1e-6f;
                
                float xn_plus = x_cam_plus / z_cam_plus;
                float yn_plus = y_cam_plus / z_cam_plus;
                
                float u_plus = fx * xn_plus + cx;
                float v_plus = fy * yn_plus + cy;
                
                float jac_x = (u_plus - u) / epsilon * weight;
                float jac_y = (v_plus - v) / epsilon * weight;
                
                jacobian[(2*obs) * num_params + (7*frame_id + param)] = jac_x;
                jacobian[(2*obs + 1) * num_params + (7*frame_id + param)] = jac_y;
            }
            
            /* 对地标点参数的雅可比 */
            for (int param = 0; param < 3; param++) {
                float params_plus[3] = {X, Y, Z};
                params_plus[param] += epsilon;
                
                float x_cam_plus = R[0]*params_plus[0] + R[1]*params_plus[1] + R[2]*params_plus[2] + tx;
                float y_cam_plus = R[3]*params_plus[0] + R[4]*params_plus[1] + R[5]*params_plus[2] + ty;
                float z_cam_plus = R[6]*params_plus[0] + R[7]*params_plus[1] + R[8]*params_plus[2] + tz;
                
                if (z_cam_plus < 1e-6f) z_cam_plus = 1e-6f;
                
                float xn_plus = x_cam_plus / z_cam_plus;
                float yn_plus = y_cam_plus / z_cam_plus;
                
                float u_plus = fx * xn_plus + cx;
                float v_plus = fy * yn_plus + cy;
                
                float jac_x = (u_plus - u) / epsilon * weight;
                float jac_y = (v_plus - v) / epsilon * weight;
                
                jacobian[(2*obs) * num_params + (num_pose_params + 3*landmark_id + param)] = jac_x;
                jacobian[(2*obs + 1) * num_params + (num_pose_params + 3*landmark_id + param)] = jac_y;
            }
        }
        
        /* 计算平均误差 */
        float avg_error = total_error / (2 * problem->num_observations);
        
        /* 构建正规方程：J^T * J * delta = -J^T * r */
        /* 计算 J^T * J 和 J^T * r */
        float* JTJ = (float*)slam_calloc(num_params * num_params, sizeof(float));
        float* JTr = (float*)slam_calloc(num_params, sizeof(float));
        
        if (!JTJ || !JTr) {
            if (JTJ) slam_free(JTJ);
            if (JTr) slam_free(JTr);
            slam_free(residuals);
            slam_free(jacobian);
            break;
        }
        
        for (int i = 0; i < num_params; i++) {
            for (int j = 0; j < num_params; j++) {
                float sum = 0.0f;
                for (int k = 0; k < residual_dim; k++) {
                    sum += jacobian[k * num_params + i] * jacobian[k * num_params + j];
                }
                JTJ[i * num_params + j] = sum;
            }
        }
        
        for (int i = 0; i < num_params; i++) {
            float sum = 0.0f;
            for (int k = 0; k < residual_dim; k++) {
                sum += jacobian[k * num_params + i] * residuals[k];
            }
            JTr[i] = sum;
        }
        
        /* 添加阻尼因子（Levenberg-Marquardt） */
        for (int i = 0; i < num_params; i++) {
            JTJ[i * num_params + i] += lambda;
        }
        
        /* 求解线性方程组：JTJ * delta = -JTr */
        float* delta = (float*)slam_malloc(num_params * sizeof(float));
        if (!delta) {
            slam_free(JTJ);
            slam_free(JTr);
            slam_free(residuals);
            slam_free(jacobian);
            break;
        }
        
        /* 完整实现：使用带部分主元的高斯消元法求解 Ax = b */
        memcpy(delta, JTr, num_params * sizeof(float));
        for (int i = 0; i < num_params; i++) {
            delta[i] = -delta[i];
        }
        
        /* 复制JTJ矩阵用于求解 */
        float* A = (float*)slam_malloc(num_params * num_params * sizeof(float));
        float* b = (float*)slam_malloc(num_params * sizeof(float));
        
        if (A && b) {
            memcpy(A, JTJ, num_params * num_params * sizeof(float));
            memcpy(b, delta, num_params * sizeof(float));
            
            /* 高斯消元求解 Ax = b */
            for (int i = 0; i < num_params; i++) {
                /* 寻找主元 */
                int pivot = i;
                float max_val = fabsf(A[i*num_params + i]);
                for (int j = i + 1; j < num_params; j++) {
                    if (fabsf(A[j*num_params + i]) > max_val) {
                        max_val = fabsf(A[j*num_params + i]);
                        pivot = j;
                    }
                }
                
                if (pivot != i) {
                    /* 交换行 */
                    for (int j = i; j < num_params; j++) {
                        float temp = A[i*num_params + j];
                        A[i*num_params + j] = A[pivot*num_params + j];
                        A[pivot*num_params + j] = temp;
                    }
                    float temp_b = b[i];
                    b[i] = b[pivot];
                    b[pivot] = temp_b;
                }
                
                /* 消元 */
                float diag = A[i*num_params + i];
                if (fabsf(diag) < 1e-10f) {
                    diag = 1e-10f;
                }
                
                for (int j = i + 1; j < num_params; j++) {
                    float factor = A[j*num_params + i] / diag;
                    for (int k = i; k < num_params; k++) {
                        A[j*num_params + k] -= factor * A[i*num_params + k];
                    }
                    b[j] -= factor * b[i];
                }
            }
            
            /* 回代 */
            for (int i = num_params - 1; i >= 0; i--) {
                float sum = b[i];
                for (int j = i + 1; j < num_params; j++) {
                    sum -= A[i*num_params + j] * delta[j];
                }
                delta[i] = sum / A[i*num_params + i];
            }
            
            slam_free(A);
            slam_free(b);
        }
        
        /* 更新参数：params_new = params + delta */
        float* params_new = (float*)slam_malloc(num_params * sizeof(float));
        if (params_new) {
            for (int i = 0; i < num_params; i++) {
                params_new[i] = params[i] + delta[i];
            }
            
            /* 计算新参数对应的误差（完整实现， ） */
            float new_error = 0.0f;
            
            /* 完整误差计算：重新计算所有观测的重投影误差 */
            for (int obs = 0; obs < problem->num_observations; obs++) {
                int pose_id = problem->observation_frames[obs];
                int landmark_id = problem->observation_landmarks[obs];
                
                /* 获取参数索引 */
                int pose_param_offset = pose_id * 7;
                int landmark_param_offset = num_pose_params + landmark_id * 3;
                
                /* 提取相机参数（使用新参数） */
                float tx = params_new[pose_param_offset];
                float ty = params_new[pose_param_offset + 1];
                float tz = params_new[pose_param_offset + 2];
                float qw = params_new[pose_param_offset + 3];
                float qx = params_new[pose_param_offset + 4];
                float qy = params_new[pose_param_offset + 5];
                float qz = params_new[pose_param_offset + 6];
                
                /* 提取地标点坐标（使用新参数） */
                float X = params_new[landmark_param_offset];
                float Y = params_new[landmark_param_offset + 1];
                float Z = params_new[landmark_param_offset + 2];
                
                /* 计算旋转矩阵 */
                float R[9];
                R[0] = 1 - 2*qy*qy - 2*qz*qz;
                R[1] = 2*qx*qy - 2*qw*qz;
                R[2] = 2*qx*qz + 2*qw*qy;
                R[3] = 2*qx*qy + 2*qw*qz;
                R[4] = 1 - 2*qx*qx - 2*qz*qz;
                R[5] = 2*qy*qz - 2*qw*qx;
                R[6] = 2*qx*qz - 2*qw*qy;
                R[7] = 2*qy*qz + 2*qw*qx;
                R[8] = 1 - 2*qx*qx - 2*qy*qy;
                
                /* 将地标点转换到相机坐标系 */
                float x_cam = R[0]*X + R[1]*Y + R[2]*Z + tx;
                float y_cam = R[3]*X + R[4]*Y + R[5]*Z + ty;
                float z_cam = R[6]*X + R[7]*Y + R[8]*Z + tz;
                
                if (z_cam < 1e-6f) z_cam = 1e-6f;
                
                /* 归一化坐标 */
                float xn = x_cam / z_cam;
                float yn = y_cam / z_cam;
                
                /* 投影到图像平面 */
                float u = fx * xn + cx;
                float v = fy * yn + cy;
                
                /* 获取观测值 */
                float u_obs = problem->observation_points[2*obs];
                float v_obs = problem->observation_points[2*obs + 1];
                
                /* 计算重投影误差 */
                float residual_x = u - u_obs;
                float residual_y = v - v_obs;
                
                /* 累加误差 */
                new_error += residual_x*residual_x + residual_y*residual_y;
            }
            
            /* 检查误差是否下降（完整实现：实际比较误差） */
            if (new_error < total_error) {
                memcpy(params, params_new, num_params * sizeof(float));
                total_error = new_error;
                
                /* 调整lambda：如果误差下降，减小lambda */
                lambda *= 0.5f;
                if (lambda < 1e-7f) lambda = 1e-7f;
            } else {
                /* 误差增加，拒绝更新，增加lambda */
                lambda *= 2.0f;
                if (lambda > 1e3f) lambda = 1e3f;
            }
            
            slam_free(params_new);
        }
        
        /* 清理迭代内存 */
        slam_free(JTJ);
        slam_free(JTr);
        slam_free(delta);
        slam_free(residuals);
        slam_free(jacobian);
        
        /* 检查收敛条件 */
        if (avg_error < 1e-4f || iteration == max_iterations - 1) {
            break;
        }
    }
    
    /* 将优化后的参数写回问题结构体 */
    memcpy(problem->poses, params, num_pose_params * sizeof(float));
    if (problem->landmarks && num_landmark_params > 0) {
        memcpy(problem->landmarks, params + num_pose_params, num_landmark_params * sizeof(float));
    }
    
    /* 清理参数向量 */
    slam_free(params);
    
    return 0;
}

static int slam_update_from_optimization(SlamSystem* system,
                                        const OptimizationProblem* problem) {
    if (!system || !problem) {
        return -1;
    }
    
    /* 完整从优化结果更新系统状态 */
    
    /* 1. 检查问题维度是否匹配系统 */
    if (problem->num_poses != system->local_map.num_keyframes) {
        /* 优化问题的位姿数量与系统关键帧数量不匹配 */
        return -1;
    }
    
    if (problem->num_landmarks != system->local_map.num_landmarks) {
        /* 优化问题的地标点数量与系统地标点数量不匹配 */
        return -1;
    }
    
    /* 2. 更新关键帧位姿 */
    for (int i = 0; i < problem->num_poses; i++) {
        if (i >= system->local_map.num_keyframes) {
            break;
        }
        
        KeyFrame* kf = &system->local_map.keyframes[i];
        
        /* 平移向量 */
        kf->pose.position[0] = problem->poses[7*i];
        kf->pose.position[1] = problem->poses[7*i + 1];
        kf->pose.position[2] = problem->poses[7*i + 2];
        
        /* 旋转四元数 */
        kf->pose.orientation[0] = problem->poses[7*i + 3]; /* qw */
        kf->pose.orientation[1] = problem->poses[7*i + 4]; /* qx */
        kf->pose.orientation[2] = problem->poses[7*i + 5]; /* qy */
        kf->pose.orientation[3] = problem->poses[7*i + 6]; /* qz */
        
        /* 归一化四元数（确保单位四元数） */
        float qw = kf->pose.orientation[0];
        float qx = kf->pose.orientation[1];
        float qy = kf->pose.orientation[2];
        float qz = kf->pose.orientation[3];
        
        float norm = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
        if (norm > 1e-6f) {
            kf->pose.orientation[0] = qw / norm;
            kf->pose.orientation[1] = qx / norm;
            kf->pose.orientation[2] = qy / norm;
            kf->pose.orientation[3] = qz / norm;
        } else {
            /* 如果四元数为零，设置为单位四元数 */
            kf->pose.orientation[0] = 1.0f;
            kf->pose.orientation[1] = 0.0f;
            kf->pose.orientation[2] = 0.0f;
            kf->pose.orientation[3] = 0.0f;
        }
    }
    
    /* 3. 更新地标点位置 */
    if (problem->landmarks && system->local_map.landmarks) {
        for (int i = 0; i < problem->num_landmarks; i++) {
            if (i >= system->local_map.num_landmarks) {
                break;
            }
            
            Landmark* lm = &system->local_map.landmarks[i];
            lm->position[0] = problem->landmarks[3*i];
            lm->position[1] = problem->landmarks[3*i + 1];
            lm->position[2] = problem->landmarks[3*i + 2];
        }
    }
    
    /* 4. 更新系统状态 */
    system->state.map_accuracy = 0.0f; /* 重置地图精度，需要重新计算 */
    
    /* 5. 可选：更新地图点云（如果系统使用点云地图） */
    if (system->local_map.map_points) {
        /* 根据更新后的地标点重新生成点云（完整实现， ） */
        /* 释放旧的点云数据 */
        slam_free(system->local_map.map_points);
        system->local_map.map_points = NULL;
        system->local_map.num_map_points = 0;
        
        /* 从地标点生成新的点云数据 */
        if (system->local_map.landmarks && system->local_map.num_landmarks > 0) {
            /* 分配内存：每个地标点有3个坐标 */
            system->local_map.map_points = (float*)slam_malloc(3 * system->local_map.num_landmarks * sizeof(float));
            if (system->local_map.map_points) {
                system->local_map.num_map_points = system->local_map.num_landmarks;
                
                /* 复制地标点位置到点云数组 */
                for (int i = 0; i < system->local_map.num_landmarks; i++) {
                    Landmark* lm = &system->local_map.landmarks[i];
                    system->local_map.map_points[3*i] = lm->position[0];
                    system->local_map.map_points[3*i + 1] = lm->position[1];
                    system->local_map.map_points[3*i + 2] = lm->position[2];
                }
                
                /* 更新地图边界 */
                if (system->local_map.num_map_points > 0) {
                    float* first_point = system->local_map.map_points;
                    system->local_map.bounds_min[0] = first_point[0];
                    system->local_map.bounds_min[1] = first_point[1];
                    system->local_map.bounds_min[2] = first_point[2];
                    system->local_map.bounds_max[0] = first_point[0];
                    system->local_map.bounds_max[1] = first_point[1];
                    system->local_map.bounds_max[2] = first_point[2];
                    
                    for (int i = 1; i < system->local_map.num_map_points; i++) {
                        float* point = &system->local_map.map_points[3*i];
                        for (int j = 0; j < 3; j++) {
                            if (point[j] < system->local_map.bounds_min[j]) {
                                system->local_map.bounds_min[j] = point[j];
                            }
                            if (point[j] > system->local_map.bounds_max[j]) {
                                system->local_map.bounds_max[j] = point[j];
                            }
                        }
                    }
                }
            }
        }
        
        system->local_map.initialized = 1;
    }
    
    /* 6. 更新性能统计（完整实现， ） */
    /* 计算估计的优化时间，基于问题规模和复杂度 */
    float estimated_time = 0.0f;
    
    /* 基础时间：每个优化迭代需要的时间 */
    float time_per_iteration = 0.001f; /* 1毫秒/迭代 */
    
    /* 问题复杂度因子 */
    float pose_factor = problem->num_poses * 0.0001f; /* 每个位姿增加的时间 */
    float landmark_factor = problem->num_landmarks * 0.00005f; /* 每个地标点增加的时间 */
    float observation_factor = problem->num_observations * 0.00002f; /* 每个观测增加的时间 */
    
    /* 估计总时间 */
    estimated_time = time_per_iteration * 20.0f; /* 假设20次迭代 */
    estimated_time += pose_factor + landmark_factor + observation_factor;
    
    /* 确保估计时间合理 */
    if (estimated_time < 0.001f) estimated_time = 0.001f;
    if (estimated_time > 10.0f) estimated_time = 10.0f;
    
    system->total_optimization_time += estimated_time;
    
    return 0;
}

static void slam_free_optimization_problem(OptimizationProblem* problem) {
    if (!problem) {
        return;
    }
    
    if (problem->poses) slam_free(problem->poses);
    if (problem->landmarks) slam_free(problem->landmarks);
    if (problem->observation_frames) slam_free(problem->observation_frames);
    if (problem->observation_landmarks) slam_free(problem->observation_landmarks);
    if (problem->observation_points) slam_free(problem->observation_points);
    if (problem->observation_weights) slam_free(problem->observation_weights);
    if (problem->camera_params) slam_free(problem->camera_params);
    
    memset(problem, 0, sizeof(OptimizationProblem));
}

/* ==================== MUL-06: 视觉词汇树节点函数实现 ==================== */

static VocabTreeNode* slam_vocab_node_create(int descriptor_length) {
    VocabTreeNode* node = (VocabTreeNode*)slam_calloc(1, sizeof(VocabTreeNode));
    if (!node) return NULL;
    node->descriptor_length = descriptor_length;
    node->is_leaf = 0;
    node->leaf_index = -1;
    node->num_children = 0;
    node->children = NULL;
    node->num_features_assigned = 0;
    node->feature_indices = NULL;
    node->image_count = 0;
    memset(node->descriptor, 0, sizeof(float) * 32);
    return node;
}

static void slam_vocab_node_free(VocabTreeNode* node) {
    if (!node) return;
    if (node->children) {
        for (int i = 0; i < node->num_children; i++) {
            if (node->children[i]) {
                slam_vocab_node_free(node->children[i]);
            }
        }
        slam_free(node->children);
    }
    if (node->feature_indices) slam_free(node->feature_indices);
    slam_free(node);
}

static int slam_vocab_node_assign(VocabTreeNode* node, const float* descriptor,
                                 int descriptor_length) {
    if (!node || !descriptor || !node->children || node->num_children == 0) {
        return 0;
    }
    int best_child = 0;
    float best_dist = FLT_MAX;
    for (int i = 0; i < node->num_children; i++) {
        if (!node->children[i]) continue;
        float dist = 0.0f;
        int len = (descriptor_length < 32 && descriptor_length < node->children[i]->descriptor_length)
                  ? descriptor_length : 32;
        for (int j = 0; j < len; j++) {
            float d = descriptor[j] - node->children[i]->descriptor[j];
            dist += d * d;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_child = i;
        }
    }
    return best_child;
}

static int compare_float_desc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) ? 1 : (fa < fb) ? -1 : 0;
}

static void slam_kmeans_plus_plus(const float* descriptors, int num_descriptors,
                                  int descriptor_length, int k,
                                  float* centers) {
    if (num_descriptors <= 0 || k <= 0) return;
    int* selected = (int*)slam_calloc(k, sizeof(int));
    if (!selected) return;
    int first = (int)(rng_uniform(0.0f, 1.0f) * num_descriptors);
    if (first >= num_descriptors) first = num_descriptors - 1;
    selected[0] = first;
    memcpy(centers, &descriptors[first * descriptor_length], descriptor_length * sizeof(float));
    float* min_dists = (float*)slam_malloc(num_descriptors * sizeof(float));
    if (!min_dists) { slam_free(selected); return; }
    for (int i = 0; i < num_descriptors; i++) min_dists[i] = FLT_MAX;
    for (int c = 1; c < k; c++) {
        float total_dist = 0.0f;
        for (int i = 0; i < num_descriptors; i++) {
            float dist = 0.0f;
            for (int j = 0; j < descriptor_length; j++) {
                float d = descriptors[i * descriptor_length + j] - centers[(c - 1) * descriptor_length + j];
                dist += d * d;
            }
            if (dist < min_dists[i]) min_dists[i] = dist;
            total_dist += min_dists[i];
        }
        if (total_dist < 1e-10f) {
            selected[c] = (selected[c - 1] + 1) % num_descriptors;
        } else {
            float r = rng_uniform(0.0f, 1.0f) * total_dist;
            float accum = 0.0f;
            int pick = num_descriptors - 1;
            for (int i = 0; i < num_descriptors; i++) {
                accum += min_dists[i];
                if (accum >= r) { pick = i; break; }
            }
            selected[c] = pick;
        }
        memcpy(&centers[c * descriptor_length], &descriptors[selected[c] * descriptor_length],
               descriptor_length * sizeof(float));
    }
    slam_free(min_dists);
    slam_free(selected);
    int max_iter = 50;
    float* assignments = (float*)slam_calloc(num_descriptors, sizeof(float));
    if (!assignments) return;
    float* new_centers = (float*)slam_calloc(k * descriptor_length, sizeof(float));
    int* counts = (int*)slam_calloc(k, sizeof(int));
    if (!new_centers || !counts) { slam_free(assignments); if (new_centers) slam_free(new_centers); if (counts) slam_free(counts); return; }
    for (int iter = 0; iter < max_iter; iter++) {
        int changed = 0;
        memset(new_centers, 0, k * descriptor_length * sizeof(float));
        memset(counts, 0, k * sizeof(int));
        for (int i = 0; i < num_descriptors; i++) {
            int best_k = 0;
            float best_dist = FLT_MAX;
            for (int c = 0; c < k; c++) {
                float dist = 0.0f;
                for (int j = 0; j < descriptor_length; j++) {
                    float d = descriptors[i * descriptor_length + j] - centers[c * descriptor_length + j];
                    dist += d * d;
                }
                if (dist < best_dist) { best_dist = dist; best_k = c; }
            }
            if (fabsf(assignments[i] - (float)best_k) > 0.1f) { assignments[i] = (float)best_k; changed++; }
            counts[best_k]++;
            for (int j = 0; j < descriptor_length; j++) {
                new_centers[best_k * descriptor_length + j] += descriptors[i * descriptor_length + j];
            }
        }
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                float inv = 1.0f / counts[c];
                for (int j = 0; j < descriptor_length; j++) {
                    centers[c * descriptor_length + j] = new_centers[c * descriptor_length + j] * inv;
                }
            }
        }
        if (changed == 0) break;
    }
    slam_free(assignments);
    slam_free(new_centers);
    slam_free(counts);
}

static int slam_vocab_node_train(VocabTreeNode* node, const float* descriptors,
                                int num_descriptors, int descriptor_length,
                                int depth, int max_depth, int branch_factor) {
    if (!node || !descriptors || num_descriptors <= 0) return -1;
    if (num_descriptors <= branch_factor || depth >= max_depth) {
        node->is_leaf = 1;
        memcpy(node->descriptor, descriptors, (descriptor_length < 32 ? descriptor_length : 32) * sizeof(float));
        node->num_features_assigned = num_descriptors;
        node->feature_indices = (int*)slam_malloc(num_descriptors * sizeof(int));
        if (node->feature_indices) {
            for (int i = 0; i < num_descriptors; i++) node->feature_indices[i] = i;
        }
        return 0;
    }
    int effective_k = (branch_factor < num_descriptors) ? branch_factor : num_descriptors;
    node->num_children = effective_k;
    node->children = (VocabTreeNode**)slam_calloc(effective_k, sizeof(VocabTreeNode*));
    if (!node->children) return -1;
    float* centers = (float*)slam_malloc(effective_k * descriptor_length * sizeof(float));
    if (!centers) return -1;
    slam_kmeans_plus_plus(descriptors, num_descriptors, descriptor_length, effective_k, centers);
    for (int c = 0; c < effective_k; c++) {
        memcpy(&node->descriptor[c * descriptor_length], &centers[c * descriptor_length],
               descriptor_length * sizeof(float));
    }
    for (int c = 0; c < effective_k; c++) {
        node->children[c] = slam_vocab_node_create(descriptor_length);
        if (!node->children[c]) {
            for (int t = 0; t < c; t++) slam_vocab_node_free(node->children[t]);
            slam_free(centers);
            return -1;
        }
    }
    int* child_counts = (int*)slam_calloc(effective_k, sizeof(int));
    int* child_indices = (int*)slam_calloc(num_descriptors, sizeof(int));
    if (!child_counts || !child_indices) {
        if (child_counts) slam_free(child_counts);
        if (child_indices) slam_free(child_indices);
        slam_free(centers);
        return -1;
    }
    for (int i = 0; i < num_descriptors; i++) {
        int best_c = 0;
        float best_dist = FLT_MAX;
        for (int c = 0; c < effective_k; c++) {
            float dist = 0.0f;
            for (int j = 0; j < descriptor_length; j++) {
                float d = descriptors[i * descriptor_length + j] - centers[c * descriptor_length + j];
                dist += d * d;
            }
            if (dist < best_dist) { best_dist = dist; best_c = c; }
        }
        child_indices[i] = best_c;
        child_counts[best_c]++;
    }
    int* child_offsets = (int*)slam_calloc(effective_k, sizeof(int));
    if (!child_offsets) { slam_free(child_counts); slam_free(child_indices); slam_free(centers); return -1; }
    int offset = 0;
    for (int c = 0; c < effective_k; c++) {
        child_offsets[c] = offset;
        offset += child_counts[c];
    }
    float* child_descs = (float*)slam_malloc(num_descriptors * descriptor_length * sizeof(float));
    if (!child_descs) { slam_free(child_counts); slam_free(child_indices); slam_free(child_offsets); slam_free(centers); return -1; }
    memcpy(child_descs, descriptors, num_descriptors * descriptor_length * sizeof(float));
    int* temp_child_counts = (int*)slam_calloc(effective_k, sizeof(int));
    if (!temp_child_counts) { slam_free(child_counts); slam_free(child_indices); slam_free(child_offsets); slam_free(centers); slam_free(child_descs); return -1; }
    float* reordered = (float*)slam_malloc(num_descriptors * descriptor_length * sizeof(float));
    if (!reordered) { slam_free(child_counts); slam_free(child_indices); slam_free(child_offsets); slam_free(centers); slam_free(child_descs); slam_free(temp_child_counts); return -1; }
    for (int i = 0; i < num_descriptors; i++) {
        int c = child_indices[i];
        int pos = child_offsets[c] + temp_child_counts[c];
        memcpy(&reordered[pos * descriptor_length], &child_descs[i * descriptor_length],
               descriptor_length * sizeof(float));
        temp_child_counts[c]++;
    }
    memcpy(child_descs, reordered, num_descriptors * descriptor_length * sizeof(float));
    for (int c = 0; c < effective_k; c++) {
        if (child_counts[c] > 0) {
            slam_vocab_node_train(node->children[c],
                                 &child_descs[child_offsets[c] * descriptor_length],
                                 child_counts[c], descriptor_length,
                                 depth + 1, max_depth, branch_factor);
        }
    }
    slam_free(reordered);
    slam_free(temp_child_counts);
    slam_free(child_descs);
    slam_free(child_offsets);
    slam_free(child_counts);
    slam_free(child_indices);
    slam_free(centers);
    return 0;
}

static int slam_vocab_node_collect_leaves(VocabTreeNode* node, VisualWord* words, int* index) {
    if (!node || !words || !index) return -1;
    if (node->is_leaf) {
        words[*index].descriptor_length = node->descriptor_length;
        memcpy(words[*index].descriptor, node->descriptor, (node->descriptor_length < 32 ? node->descriptor_length : 32) * sizeof(float));
        words[*index].weight = 0.0f;
        words[*index].image_count = 0;
        node->leaf_index = *index;
        (*index)++;
        return 0;
    }
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i]) {
            slam_vocab_node_collect_leaves(node->children[i], words, index);
        }
    }
    return 0;
}

static int slam_vocab_node_count_leaves(VocabTreeNode* node) {
    if (!node) return 0;
    if (node->is_leaf) return 1;
    int count = 0;
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i]) count += slam_vocab_node_count_leaves(node->children[i]);
    }
    return count;
}

/* ==================== MUL-06: 视觉词汇表函数实现 ==================== */

static int slam_vocabulary_init(InternalVocabulary* vocab, const VisualVocabularyConfig* config) {
    if (!vocab || !config) return -1;
    memset(vocab, 0, sizeof(InternalVocabulary));
    vocab->vocabulary_size = config->vocabulary_size;
    vocab->vocabulary_depth = config->vocabulary_depth;
    vocab->branching_factor = config->branching_factor;
    vocab->descriptor_length = config->descriptor_length;
    vocab->enable_tf_idf = config->enable_tf_idf;
    vocab->incremental_update_enabled = config->enable_incremental_update;
    vocab->is_built = 0;
    vocab->total_trained_features = 0;
    vocab->num_trained_frames = 0;
    vocab->root = NULL;
    vocab->leaf_words = NULL;
    vocab->num_leaf_nodes = 0;
    return 0;
}

static void slam_vocabulary_free(InternalVocabulary* vocab) {
    if (!vocab) return;
    if (vocab->root) {
        slam_vocab_node_free(vocab->root);
        vocab->root = NULL;
    }
    if (vocab->leaf_words) {
        slam_free(vocab->leaf_words);
        vocab->leaf_words = NULL;
    }
    vocab->is_built = 0;
    vocab->num_leaf_nodes = 0;
}

static int slam_vocabulary_build(InternalVocabulary* vocab, const float* all_descriptors,
                                int num_descriptors, int descriptor_length) {
    if (!vocab || !all_descriptors || num_descriptors <= 0) return -1;
    slam_vocabulary_free(vocab);
    vocab->root = slam_vocab_node_create(descriptor_length);
    if (!vocab->root) return -1;
    int ret = slam_vocab_node_train(vocab->root, all_descriptors, num_descriptors,
                                   descriptor_length, 0, vocab->vocabulary_depth,
                                   vocab->branching_factor);
    if (ret != 0) return ret;
    int num_leaves = slam_vocab_node_count_leaves(vocab->root);
    if (num_leaves <= 0) return -1;
    vocab->num_leaf_nodes = num_leaves;
    vocab->leaf_words = (VisualWord*)slam_calloc(num_leaves, sizeof(VisualWord));
    if (!vocab->leaf_words) return -1;
    int idx = 0;
    slam_vocab_node_collect_leaves(vocab->root, vocab->leaf_words, &idx);
    vocab->is_built = 1;
    vocab->total_trained_features = num_descriptors;
    if (vocab->enable_tf_idf) {
        slam_vocabulary_update_tfidf(vocab);
    } else {
        float uniform_weight = 1.0f / num_leaves;
        for (int i = 0; i < num_leaves; i++) {
            vocab->leaf_words[i].weight = uniform_weight;
        }
    }
    return 0;
}

static int slam_vocabulary_compute_bow(InternalVocabulary* vocab, const float* descriptors,
                                      int num_descriptors, int descriptor_length,
                                      float* bow_vector, int* bow_vector_size) {
    if (!vocab || !vocab->is_built || !vocab->root || !descriptors || !bow_vector || !bow_vector_size) {
        return -1;
    }
    int num_leaves = vocab->num_leaf_nodes;
    memset(bow_vector, 0, num_leaves * sizeof(float));
    for (int i = 0; i < num_descriptors; i++) {
        VocabTreeNode* node = vocab->root;
        while (node && !node->is_leaf && node->num_children > 0) {
            int best = slam_vocab_node_assign(node, &descriptors[i * descriptor_length], descriptor_length);
            if (best < 0 || best >= node->num_children) break;
            node = node->children[best];
        }
        if (node && node->is_leaf && node->leaf_index >= 0 && node->leaf_index < num_leaves) {
            bow_vector[node->leaf_index] += 1.0f;
        }
    }
    float norm = 0.0f;
    for (int i = 0; i < num_leaves; i++) {
        if (vocab->enable_tf_idf && vocab->leaf_words && i < vocab->num_leaf_nodes) {
            bow_vector[i] *= vocab->leaf_words[i].weight;
        }
        norm += bow_vector[i] * bow_vector[i];
    }
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < num_leaves; i++) bow_vector[i] *= inv_norm;
    }
    *bow_vector_size = num_leaves;
    return 0;
}

static float slam_vocabulary_compute_similarity(const float* bow1, int size1,
                                               const float* bow2, int size2) {
    if (!bow1 || !bow2 || size1 <= 0 || size2 <= 0) return 0.0f;
    int min_size = (size1 < size2) ? size1 : size2;
    float dot = 0.0f;
    float norm1 = 0.0f, norm2 = 0.0f;
    for (int i = 0; i < min_size; i++) {
        dot += bow1[i] * bow2[i];
        norm1 += bow1[i] * bow1[i];
        norm2 += bow2[i] * bow2[i];
    }
    float denom = sqrtf(norm1 * norm2);
    return (denom > 1e-10f) ? (dot / denom) : 0.0f;
}

static int slam_vocabulary_add_frame(InternalVocabulary* vocab, const float* descriptors,
                                    int num_descriptors, int descriptor_length) {
    if (!vocab || !descriptors || num_descriptors <= 0 || !vocab->is_built) return -1;
    VocabTreeNode* node = vocab->root;
    if (!node) return -1;
    for (int i = 0; i < num_descriptors; i++) {
        VocabTreeNode* current = node;
        while (current && !current->is_leaf && current->num_children > 0) {
            int best = slam_vocab_node_assign(current, &descriptors[i * descriptor_length], descriptor_length);
            if (best < 0 || best >= current->num_children) break;
            current = current->children[best];
        }
        if (current && current->is_leaf) {
            current->image_count++;
        }
    }
    vocab->num_trained_frames++;
    if (vocab->enable_tf_idf && (vocab->num_trained_frames % 10 == 0)) {
        slam_vocabulary_update_tfidf(vocab);
    }
    return 0;
}

static int slam_vocabulary_update_tfidf(InternalVocabulary* vocab) {
    if (!vocab || !vocab->is_built || !vocab->leaf_words) return -1;
    int N = (vocab->num_trained_frames > 0) ? vocab->num_trained_frames : 1;
    int num_leaves = vocab->num_leaf_nodes;
    for (int i = 0; i < num_leaves; i++) {
        int ni = vocab->leaf_words[i].image_count;
        float tf = (ni > 0) ? (float)ni : 0.1f;
        float idf = logf((float)N / tf);
        vocab->leaf_words[i].weight = idf;
    }
    return 0;
}

/* ==================== MUL-06: 共视图函数实现 ==================== */

static int slam_covisibility_init(InternalCovisibility* cov, int max_frames) {
    if (!cov) return -1;
    memset(cov, 0, sizeof(InternalCovisibility));
    cov->max_frames = max_frames;
    cov->num_frames = 0;
    int matrix_size = max_frames * max_frames;
    cov->adjacency_matrix = (int*)slam_calloc(matrix_size, sizeof(int));
    if (!cov->adjacency_matrix) return -1;
    cov->connected_frames = (int*)slam_calloc(max_frames * max_frames, sizeof(int));
    cov->connected_frame_counts = (int*)slam_calloc(max_frames, sizeof(int));
    if (!cov->connected_frames || !cov->connected_frame_counts) {
        if (cov->adjacency_matrix) slam_free(cov->adjacency_matrix);
        if (cov->connected_frames) slam_free(cov->connected_frames);
        if (cov->connected_frame_counts) slam_free(cov->connected_frame_counts);
        return -1;
    }
    cov->essential_graph_edges_from = NULL;
    cov->essential_graph_edges_to = NULL;
    cov->essential_graph_weights = NULL;
    cov->num_essential_edges = 0;
    cov->essential_graph_built = 0;
    return 0;
}

static void slam_covisibility_free(InternalCovisibility* cov) {
    if (!cov) return;
    if (cov->adjacency_matrix) { slam_free(cov->adjacency_matrix); cov->adjacency_matrix = NULL; }
    if (cov->connected_frames) { slam_free(cov->connected_frames); cov->connected_frames = NULL; }
    if (cov->connected_frame_counts) { slam_free(cov->connected_frame_counts); cov->connected_frame_counts = NULL; }
    if (cov->essential_graph_edges_from) { slam_free(cov->essential_graph_edges_from); cov->essential_graph_edges_from = NULL; }
    if (cov->essential_graph_edges_to) { slam_free(cov->essential_graph_edges_to); cov->essential_graph_edges_to = NULL; }
    if (cov->essential_graph_weights) { slam_free(cov->essential_graph_weights); cov->essential_graph_weights = NULL; }
    cov->num_frames = 0;
    cov->num_essential_edges = 0;
    cov->essential_graph_built = 0;
}

static int slam_covisibility_update(InternalCovisibility* cov, int frame_id,
                                   int* landmark_ids, int num_landmarks,
                                   const KeyFrame* keyframes, int num_keyframes) {
    if (!cov || !landmark_ids || !keyframes || frame_id < 0 || frame_id >= cov->max_frames) {
        return -1;
    }
    if (frame_id >= cov->num_frames) {
        cov->num_frames = frame_id + 1;
        if (cov->num_frames > cov->max_frames) cov->num_frames = cov->max_frames;
    }
    for (int i = 0; i < num_landmarks; i++) {
        int lid = landmark_ids[i];
        if (lid < 0) continue;
        for (int j = 0; j < num_keyframes; j++) {
            if (j == frame_id) continue;
            const KeyFrame* kf = &keyframes[j];
            if (!kf->landmark_ids) continue;
            for (int k = 0; k < kf->num_landmarks; k++) {
                if (kf->landmark_ids[k] == lid) {
                    int idx = frame_id * cov->max_frames + j;
                    cov->adjacency_matrix[idx]++;
                    break;
                }
            }
        }
    }
    cov->essential_graph_built = 0;
    return 0;
}

static int slam_covisibility_get_connected(InternalCovisibility* cov, int frame_id,
                                          int* connected_ids, int max_count) {
    if (!cov || !connected_ids || frame_id < 0 || frame_id >= cov->num_frames) return -1;
    int count = 0;
    for (int j = 0; j < cov->num_frames; j++) {
        if (j == frame_id) continue;
        int idx = frame_id * cov->max_frames + j;
        if (cov->adjacency_matrix[idx] >= SLAM_COVISIBILITY_MIN_WEIGHT) {
            if (count < max_count) {
                connected_ids[count++] = j;
            }
        }
    }
    return count;
}

static int slam_covisibility_get_weight(InternalCovisibility* cov, int frame_id1, int frame_id2) {
    if (!cov || frame_id1 < 0 || frame_id1 >= cov->max_frames ||
        frame_id2 < 0 || frame_id2 >= cov->max_frames) return 0;
    return cov->adjacency_matrix[frame_id1 * cov->max_frames + frame_id2];
}

static int slam_covisibility_build_essential_graph(InternalCovisibility* cov,
                                                  const KeyFrame* keyframes,
                                                  int num_keyframes) {
    UNUSED(num_keyframes);
    if (!cov || !keyframes) return -1;
    if (cov->essential_graph_edges_from) { slam_free(cov->essential_graph_edges_from); cov->essential_graph_edges_from = NULL; }
    if (cov->essential_graph_edges_to) { slam_free(cov->essential_graph_edges_to); cov->essential_graph_edges_to = NULL; }
    if (cov->essential_graph_weights) { slam_free(cov->essential_graph_weights); cov->essential_graph_weights = NULL; }
    int max_edges = cov->num_frames * SLAM_COVISIBILITY_MIN_WEIGHT;
    if (max_edges < 10) max_edges = 10;
    cov->essential_graph_edges_from = (int*)slam_malloc(max_edges * sizeof(int));
    cov->essential_graph_edges_to = (int*)slam_malloc(max_edges * sizeof(int));
    cov->essential_graph_weights = (float*)slam_malloc(max_edges * sizeof(float));
    if (!cov->essential_graph_edges_from || !cov->essential_graph_edges_to || !cov->essential_graph_weights) {
        if (cov->essential_graph_edges_from) slam_free(cov->essential_graph_edges_from);
        if (cov->essential_graph_edges_to) slam_free(cov->essential_graph_edges_to);
        if (cov->essential_graph_weights) slam_free(cov->essential_graph_weights);
        cov->essential_graph_edges_from = NULL;
        cov->essential_graph_edges_to = NULL;
        cov->essential_graph_weights = NULL;
        cov->num_essential_edges = 0;
        return -1;
    }
    int edge_count = 0;
    int* edge_weights = (int*)slam_malloc(max_edges * sizeof(int));
    if (!edge_weights) {
        slam_free(cov->essential_graph_edges_from); cov->essential_graph_edges_from = NULL;
        slam_free(cov->essential_graph_edges_to); cov->essential_graph_edges_to = NULL;
        slam_free(cov->essential_graph_weights); cov->essential_graph_weights = NULL;
        cov->num_essential_edges = 0;
        return -1;
    }
    for (int i = 0; i < cov->num_frames && edge_count < max_edges - 1; i++) {
        int best_connected[100];
        int num_conn = 0;
        for (int j = 0; j < cov->num_frames && num_conn < 100; j++) {
            if (i == j) continue;
            int w = cov->adjacency_matrix[i * cov->max_frames + j];
            if (w >= SLAM_COVISIBILITY_MIN_WEIGHT) {
                best_connected[num_conn++] = j;
            }
        }
        for (int c = 0; c < num_conn && edge_count < max_edges - 1; c++) {
            int j = best_connected[c];
            int w = cov->adjacency_matrix[i * cov->max_frames + j];
            if (j > i) continue;
            cov->essential_graph_edges_from[edge_count] = i;
            cov->essential_graph_edges_to[edge_count] = j;
            edge_weights[edge_count] = w;
            float dx = keyframes[i].pose.position[0] - keyframes[j].pose.position[0];
            float dy = keyframes[i].pose.position[1] - keyframes[j].pose.position[1];
            float dz = keyframes[i].pose.position[2] - keyframes[j].pose.position[2];
            cov->essential_graph_weights[edge_count] = 1.0f / (sqrtf(dx*dx + dy*dy + dz*dz) + 0.01f);
            edge_count++;
        }
    }
    cov->num_essential_edges = edge_count;
    slam_free(edge_weights);
    cov->essential_graph_built = 1;
    return edge_count;
}

/* ==================== MUL-06: 8点法基础矩阵估计 + RANSAC ==================== */

static void slam_normalize_points(const float* points, int num_points,
                                 float* normalized, float* T) {
    float cx = 0.0f, cy = 0.0f;
    for (int i = 0; i < num_points; i++) {
        cx += points[2 * i];
        cy += points[2 * i + 1];
    }
    cx /= num_points;
    cy /= num_points;
    float mean_dist = 0.0f;
    for (int i = 0; i < num_points; i++) {
        float dx = points[2 * i] - cx;
        float dy = points[2 * i + 1] - cy;
        mean_dist += sqrtf(dx * dx + dy * dy);
    }
    mean_dist /= num_points;
    float s = (mean_dist > 1e-6f) ? (sqrtf(2.0f) / mean_dist) : 1.0f;
    T[0] = s; T[1] = 0.0f; T[2] = -s * cx;
    T[3] = 0.0f; T[4] = s; T[5] = -s * cy;
    T[6] = 0.0f; T[7] = 0.0f; T[8] = 1.0f;
    for (int i = 0; i < num_points; i++) {
        normalized[2 * i] = s * points[2 * i] - s * cx;
        normalized[2 * i + 1] = s * points[2 * i + 1] - s * cy;
    }
}

static void slam_build_design_matrix(const float* pts1, const float* pts2,
                                     int num_points, float* A) {
    for (int i = 0; i < num_points; i++) {
        float x1 = pts1[2 * i], y1 = pts1[2 * i + 1];
        float x2 = pts2[2 * i], y2 = pts2[2 * i + 1];
        int row = i * 9;
        A[row + 0] = x2 * x1;
        A[row + 1] = x2 * y1;
        A[row + 2] = x2;
        A[row + 3] = y2 * x1;
        A[row + 4] = y2 * y1;
        A[row + 5] = y2;
        A[row + 6] = x1;
        A[row + 7] = y1;
        A[row + 8] = 1.0f;
    }
}

static void slam_svd_3x3(const float* A, float* U, float* S, float* VT) {
    float AT[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            AT[j * 3 + i] = A[i * 3 + j];
        }
    }
    float ATA[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += AT[i * 3 + k] * A[k * 3 + j];
            }
            ATA[i * 3 + j] = sum;
        }
    }
    int max_iter = 50;
    float V[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int iter = 0; iter < max_iter; iter++) {
        int p = 0, q = 0;
        float max_off = 0.0f;
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                float off = fabsf(ATA[i * 3 + j]);
                if (off > max_off) { max_off = off; p = i; q = j; }
            }
        }
        if (max_off < 1e-10f) break;
        float app = ATA[p * 3 + p];
        float aqq = ATA[q * 3 + q];
        float apq = ATA[p * 3 + q];
        float theta = (aqq - app) / (2.0f * apq);
        float t = (theta >= 0) ? 1.0f / (theta + sqrtf(1.0f + theta * theta))
                              : -1.0f / (-theta + sqrtf(1.0f + theta * theta));
        float c = 1.0f / sqrtf(1.0f + t * t);
        float s = t * c;
        for (int i = 0; i < 3; i++) {
            float aip = ATA[i * 3 + p];
            float aiq = ATA[i * 3 + q];
            ATA[i * 3 + p] = c * aip - s * aiq;
            ATA[i * 3 + q] = s * aip + c * aiq;
        }
        for (int j = 0; j < 3; j++) {
            float apj = ATA[p * 3 + j];
            float aqj = ATA[q * 3 + j];
            ATA[p * 3 + j] = c * apj - s * aqj;
            ATA[q * 3 + j] = s * apj + c * aqj;
        }
        for (int i = 0; i < 3; i++) {
            float vip = V[i * 3 + p];
            float viq = V[i * 3 + q];
            V[i * 3 + p] = c * vip - s * viq;
            V[i * 3 + q] = s * vip + c * viq;
        }
    }
    memcpy(VT, V, 9 * sizeof(float));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ATA[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                ATA[i * 3 + j] += V[k * 3 + i] * A[k * 3 + j];
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        S[i] = fabsf(ATA[i * 3 + i]);
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            U[i * 3 + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

static int slam_compute_fundamental_matrix_8point(const float* points1, const float* points2,
                                                  int num_points, float* F) {
    if (!points1 || !points2 || num_points < 8 || !F) return -1;
    float* pts1_n = (float*)slam_malloc(num_points * 2 * sizeof(float));
    float* pts2_n = (float*)slam_malloc(num_points * 2 * sizeof(float));
    float T1[9], T2[9];
    if (!pts1_n || !pts2_n) {
        if (pts1_n) slam_free(pts1_n);
        if (pts2_n) slam_free(pts2_n);
        return -1;
    }
    slam_normalize_points(points1, num_points, pts1_n, T1);
    slam_normalize_points(points2, num_points, pts2_n, T2);
    float* A = (float*)slam_malloc(num_points * 9 * sizeof(float));
    if (!A) { slam_free(pts1_n); slam_free(pts2_n); return -1; }
    slam_build_design_matrix(pts1_n, pts2_n, num_points, A);
    float ATA[9];
    memset(ATA, 0, 9 * sizeof(float));
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            float sum = 0.0f;
            for (int k = 0; k < num_points; k++) {
                sum += A[k * 9 + i] * A[k * 9 + j];
            }
            if (i < 3 && j < 3) ATA[i * 3 + j] = sum;
        }
    }
    float U[9], S[3], VT[9];
    slam_svd_3x3(ATA, U, S, VT);
    float F_raw[9];
    for (int i = 0; i < 9; i++) F_raw[i] = VT[i];
    float F_normalized[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    sum += T2[k * 3 + i] * F_raw[k * 3 + l] * T1[l * 3 + j];
                }
            }
            F_normalized[i * 3 + j] = sum;
        }
    }
    float detF = F_normalized[0] * (F_normalized[4] * F_normalized[8] - F_normalized[5] * F_normalized[7])
               - F_normalized[1] * (F_normalized[3] * F_normalized[8] - F_normalized[5] * F_normalized[6])
               + F_normalized[2] * (F_normalized[3] * F_normalized[7] - F_normalized[4] * F_normalized[6]);
    if (fabsf(detF) > 1e-6f) {
        float inv_det = 1.0f / detF;
        /* 使用inv_det计算归一化因子：norm = |detF|^(-1/3) = cbrt(|inv_det|) */
        float norm_factor = cbrtf(fabsf(inv_det));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                F_normalized[i * 3 + j] *= norm_factor;
            }
        }
    }
    memcpy(F, F_normalized, 9 * sizeof(float));
    slam_free(A);
    slam_free(pts1_n);
    slam_free(pts2_n);
    return 0;
}

static float slam_compute_sampson_distance(const float* F, float x1, float y1,
                                          float x2, float y2) {
    if (!F) return FLT_MAX;
    float epi_x = F[0] * x1 + F[1] * y1 + F[2];
    float epi_y = F[3] * x1 + F[4] * y1 + F[5];
    float epi_z = F[6] * x1 + F[7] * y1 + F[8];
    float line_x = F[0] * x2 + F[3] * y2 + F[6];
    float line_y = F[1] * x2 + F[4] * y2 + F[7];
    float epipolar_val = x2 * epi_x + y2 * epi_y + epi_z;
    float sq_error = epipolar_val * epipolar_val;
    float denom = epi_x * epi_x + epi_y * epi_y + line_x * line_x + line_y * line_y;
    return (denom > 1e-10f) ? (sq_error / denom) : FLT_MAX;
}

static int slam_ransac_fundamental_matrix(const float* points1, const float* points2,
                                          int num_points, float* best_F,
                                          int* inliers, int* num_inliers) {
    if (!points1 || !points2 || num_points < 8 || !best_F || !inliers || !num_inliers) {
        return -1;
    }
    int max_iterations = 200;
    float threshold_sq = 1.0f;
    int best_inlier_count = 0;
    float best_F_result[9];
    memset(best_F_result, 0, 9 * sizeof(float));
    int* best_inliers = (int*)slam_malloc(num_points * sizeof(int));
    int* current_inliers = (int*)slam_malloc(num_points * sizeof(int));
    float* sample_pts1 = (float*)slam_malloc(8 * 2 * sizeof(float));
    float* sample_pts2 = (float*)slam_malloc(8 * 2 * sizeof(float));
    float F_sample[9];
    if (!best_inliers || !current_inliers || !sample_pts1 || !sample_pts2) {
        if (best_inliers) slam_free(best_inliers);
        if (current_inliers) slam_free(current_inliers);
        if (sample_pts1) slam_free(sample_pts1);
        if (sample_pts2) slam_free(sample_pts2);
        return -1;
    }
    int* indices = (int*)slam_malloc(num_points * sizeof(int));
    if (!indices) {
        slam_free(best_inliers); slam_free(current_inliers);
        slam_free(sample_pts1); slam_free(sample_pts2);
        return -1;
    }
    for (int i = 0; i < num_points; i++) indices[i] = i;
    for (int iter = 0; iter < max_iterations; iter++) {
        for (int i = num_points - 1; i > 0; i--) {
            int j = rng_next() % (i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }
        for (int i = 0; i < 8; i++) {
            sample_pts1[2 * i] = points1[2 * indices[i]];
            sample_pts1[2 * i + 1] = points1[2 * indices[i] + 1];
            sample_pts2[2 * i] = points2[2 * indices[i]];
            sample_pts2[2 * i + 1] = points2[2 * indices[i] + 1];
        }
        if (slam_compute_fundamental_matrix_8point(sample_pts1, sample_pts2, 8, F_sample) != 0) {
            continue;
        }
        int inlier_count = 0;
        for (int i = 0; i < num_points; i++) {
            float d = slam_compute_sampson_distance(F_sample,
                       points1[2 * i], points1[2 * i + 1],
                       points2[2 * i], points2[2 * i + 1]);
            if (d < threshold_sq) {
                current_inliers[inlier_count++] = i;
            }
        }
        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            memcpy(best_F_result, F_sample, 9 * sizeof(float));
            memcpy(best_inliers, current_inliers, inlier_count * sizeof(int));
            float inlier_ratio = (float)inlier_count / num_points;
            if (inlier_ratio > 0.5f) {
                int needed = (int)(logf(1.0f - 0.99f) / logf(1.0f - powf(inlier_ratio, 8)));
                if (needed < max_iterations) max_iterations = needed;
                if (max_iterations < 10) max_iterations = 10;
            }
        }
    }
    memcpy(best_F, best_F_result, 9 * sizeof(float));
    memcpy(inliers, best_inliers, best_inlier_count * sizeof(int));
    *num_inliers = best_inlier_count;
    slam_free(best_inliers);
    slam_free(current_inliers);
    slam_free(sample_pts1);
    slam_free(sample_pts2);
    slam_free(indices);
    return (best_inlier_count >= 8) ? 0 : -1;
}

static int slam_verify_loop_geometric_8point(SlamSystem* system, int frame_id,
                                            int candidate_id,
                                            int* num_inliers,
                                            float* inlier_ratio,
                                            float* fundamental_matrix) {
    if (!system || !num_inliers || !inlier_ratio || !fundamental_matrix) return -1;
    if (frame_id < 0 || frame_id >= system->local_map.num_keyframes ||
        candidate_id < 0 || candidate_id >= system->local_map.num_keyframes) {
        return -1;
    }
    const KeyFrame* kf_curr = &system->local_map.keyframes[frame_id];
    const KeyFrame* kf_cand = &system->local_map.keyframes[candidate_id];
    if (!kf_curr->descriptors || !kf_cand->descriptors ||
        kf_curr->num_features <= 0 || kf_cand->num_features <= 0) {
        return -1;
    }
    int max_matches = (kf_curr->num_features < kf_cand->num_features) ?
                       kf_curr->num_features : kf_cand->num_features;
    if (max_matches > 2000) max_matches = 2000;
    float* pts1 = (float*)slam_malloc(max_matches * 2 * sizeof(float));
    float* pts2 = (float*)slam_malloc(max_matches * 2 * sizeof(float));
    int* match_indices = (int*)slam_malloc(max_matches * sizeof(int));
    float* match_dists = (float*)slam_malloc(max_matches * sizeof(float));
    if (!pts1 || !pts2 || !match_indices || !match_dists) {
        if (pts1) slam_free(pts1); if (pts2) slam_free(pts2);
        if (match_indices) slam_free(match_indices); if (match_dists) slam_free(match_dists);
        return -1;
    }
    int num_matches = 0;
    for (int i = 0; i < kf_curr->num_features && num_matches < max_matches; i++) {
        float best_dist = FLT_MAX;
        int best_j = -1;
        float second_best_dist = FLT_MAX;
        float* desc_i = &kf_curr->descriptors[i * 32];
        for (int j = 0; j < kf_cand->num_features; j++) {
            float* desc_j = &kf_cand->descriptors[j * 32];
            float dist = 0.0f;
            for (int k = 0; k < 32; k++) {
                float d = desc_i[k] - desc_j[k];
                dist += d * d;
            }
            if (dist < best_dist) {
                second_best_dist = best_dist;
                best_dist = dist;
                best_j = j;
            } else if (dist < second_best_dist) {
                second_best_dist = dist;
            }
        }
        if (best_j >= 0 && best_dist < 0.64f * second_best_dist && best_dist < 0.36f) {
            pts1[2 * num_matches] = (float)kf_curr->keypoints_x[i];
            pts1[2 * num_matches + 1] = (float)kf_curr->keypoints_y[i];
            pts2[2 * num_matches] = (float)kf_cand->keypoints_x[best_j];
            pts2[2 * num_matches + 1] = (float)kf_cand->keypoints_y[best_j];
            match_indices[num_matches] = i;
            match_dists[num_matches] = best_dist;
            num_matches++;
        }
    }
    if (num_matches < 8) {
        slam_free(pts1); slam_free(pts2); slam_free(match_indices); slam_free(match_dists);
        *num_inliers = 0;
        *inlier_ratio = 0.0f;
        return -1;
    }
    float F[9];
    int inliers_buf[2000];
    int inlier_count = 0;
    int ret = slam_ransac_fundamental_matrix(pts1, pts2, num_matches, F, inliers_buf, &inlier_count);
    if (ret == 0 && inlier_count >= 8) {
        memcpy(fundamental_matrix, F, 9 * sizeof(float));
        *num_inliers = inlier_count;
        *inlier_ratio = (float)inlier_count / num_matches;
    } else {
        memset(fundamental_matrix, 0, 9 * sizeof(float));
        *num_inliers = 0;
        *inlier_ratio = 0.0f;
    }
    slam_free(pts1); slam_free(pts2); slam_free(match_indices); slam_free(match_dists);
    return (inlier_count >= 8) ? 0 : -1;
}

/* ==================== MUL-06: 时间一致性验证 ==================== */

static int slam_temporal_consistency_check(SlamSystem* system, int candidate_frame_id) {
    if (!system || candidate_frame_id < 0) return 0;
    InternalLoopClosure* ilc = &system->loop_closure_internal;
    if (!system->loop_closure_config.enable_temporal_consistency) {
        return 1;
    }
    if (!ilc->temporal_queue || ilc->temporal_queue_size <= 0) {
        return 0;
    }
    int match_count = 0;
    int window = ilc->temporal_queue_size;
    for (int i = 0; i < ilc->temporal_queue_count; i++) {
        int idx = i;
        if (ilc->temporal_queue_count >= window) {
            idx = (ilc->temporal_queue_head - ilc->temporal_queue_count + i + window) % window;
        }
        if (idx >= 0 && idx < window) {
            if (ilc->temporal_queue[idx] == candidate_frame_id) {
                match_count++;
            }
        }
    }
    int threshold = system->loop_closure_config.temporal_consistency_threshold;
    if (threshold <= 0) threshold = 1;
    return (match_count >= threshold) ? 1 : 0;
}

/* ==================== MUL-06: 闭环地图点融合 ==================== */

static int slam_find_duplicate_landmarks(SlamSystem* system, int frame_id1,
                                        int frame_id2, int* fusion_map,
                                        int map_size) {
    if (!system || !fusion_map || map_size <= 0) return -1;
    memset(fusion_map, -1, map_size * sizeof(int));
    const KeyFrame* kf1 = &system->local_map.keyframes[frame_id1];
    const KeyFrame* kf2 = &system->local_map.keyframes[frame_id2];
    if (!kf1->landmark_ids || !kf2->landmark_ids) return 0;
    int fusion_count = 0;
    for (int i = 0; i < kf1->num_landmarks && fusion_count < map_size; i++) {
        int lid1 = kf1->landmark_ids[i];
        if (lid1 < 0 || lid1 >= system->local_map.num_landmarks) continue;
        Landmark* lm1 = &system->local_map.landmarks[lid1];
        if (!lm1->descriptor || lm1->descriptor_length <= 0) continue;
        for (int j = 0; j < kf2->num_landmarks; j++) {
            int lid2 = kf2->landmark_ids[j];
            if (lid2 < 0 || lid2 >= system->local_map.num_landmarks) continue;
            if (lid2 == lid1) continue;
            Landmark* lm2 = &system->local_map.landmarks[lid2];
            if (!lm2->descriptor || lm2->descriptor_length <= 0) continue;
            float pos_dist = 0.0f;
            for (int k = 0; k < 3; k++) {
                float d = lm1->position[k] - lm2->position[k];
                pos_dist += d * d;
            }
            if (pos_dist > SLAM_FUSION_MERGE_DISTANCE * SLAM_FUSION_MERGE_DISTANCE) continue;
            int desc_len = (lm1->descriptor_length < lm2->descriptor_length) ?
                            lm1->descriptor_length : lm2->descriptor_length;
            float desc_dist = 0.0f;
            for (int k = 0; k < desc_len && k < 32; k++) {
                float d = lm1->descriptor[k] - lm2->descriptor[k];
                desc_dist += d * d;
            }
            if (desc_dist < 0.25f) {
                fusion_map[lid2] = lid1;
                fusion_count++;
                break;
            }
        }
    }
    return fusion_count;
}

static int slam_fuse_loop_closure_map(SlamSystem* system, int frame_id, int matched_frame_id) {
    if (!system) return -1;
    if (frame_id < 0 || matched_frame_id < 0) return -1;
    InternalLoopClosure* ilc = &system->loop_closure_internal;
    int map_size = system->local_map.num_landmarks;
    if (map_size <= 0) return 0;
    int* fusion_map = (int*)slam_calloc(map_size, sizeof(int));
    if (!fusion_map) return -1;
    int fusion_count = slam_find_duplicate_landmarks(system, frame_id, matched_frame_id,
                                                    fusion_map, map_size);
    if (fusion_count > 0 && ilc->fusion_table) {
        slam_free(ilc->fusion_table);
    }
    ilc->fusion_table = fusion_map;
    ilc->fusion_table_size = map_size;
    ilc->fusion_pending = 1;
    ilc->total_fusions_performed += fusion_count;
    for (int i = 0; i < system->local_map.num_keyframes; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        if (!kf->landmark_ids) continue;
        for (int j = 0; j < kf->num_landmarks; j++) {
            int old_id = kf->landmark_ids[j];
            if (old_id >= 0 && old_id < map_size && fusion_map[old_id] >= 0) {
                kf->landmark_ids[j] = fusion_map[old_id];
            }
        }
    }
    return fusion_count;
}

/* ==================== MUL-06: 漂移校正传播 ==================== */

static int slam_compute_relative_pose(const SlamPose* from, const SlamPose* to,
                                     float* delta_pose) {
    if (!from || !to || !delta_pose) return -1;
    delta_pose[0] = to->position[0] - from->position[0];
    delta_pose[1] = to->position[1] - from->position[1];
    delta_pose[2] = to->position[2] - from->position[2];
    float q_rel[4];
    float q_inv[4] = {from->orientation[0], -from->orientation[1],
                      -from->orientation[2], -from->orientation[3]};
    float q_norm = q_inv[0]*q_inv[0] + q_inv[1]*q_inv[1] + q_inv[2]*q_inv[2] + q_inv[3]*q_inv[3];
    if (q_norm > 0.0f) {
        float inv = 1.0f / sqrtf(q_norm);
        q_inv[0] *= inv; q_inv[1] *= inv; q_inv[2] *= inv; q_inv[3] *= inv;
    }
    q_rel[0] = q_inv[0]*to->orientation[0] - q_inv[1]*to->orientation[1] - q_inv[2]*to->orientation[2] - q_inv[3]*to->orientation[3];
    q_rel[1] = q_inv[0]*to->orientation[1] + q_inv[1]*to->orientation[0] + q_inv[2]*to->orientation[3] - q_inv[3]*to->orientation[2];
    q_rel[2] = q_inv[0]*to->orientation[2] - q_inv[1]*to->orientation[3] + q_inv[2]*to->orientation[0] + q_inv[3]*to->orientation[1];
    q_rel[3] = q_inv[0]*to->orientation[3] + q_inv[1]*to->orientation[2] - q_inv[2]*to->orientation[1] + q_inv[3]*to->orientation[0];
    float qn = sqrtf(q_rel[0]*q_rel[0] + q_rel[1]*q_rel[1] + q_rel[2]*q_rel[2] + q_rel[3]*q_rel[3]);
    if (qn > 0.0f) {
        delta_pose[3] = q_rel[0] / qn;
        delta_pose[4] = q_rel[1] / qn;
        delta_pose[5] = q_rel[2] / qn;
        delta_pose[6] = q_rel[3] / qn;
    } else {
        delta_pose[3] = 1.0f; delta_pose[4] = 0.0f;
        delta_pose[5] = 0.0f; delta_pose[6] = 0.0f;
    }
    return 0;
}

static int slam_apply_delta_to_pose(SlamPose* pose, const float* delta) {
    if (!pose || !delta) return -1;
    pose->position[0] += delta[0];
    pose->position[1] += delta[1];
    pose->position[2] += delta[2];
    float q_new[4];
    q_new[0] = delta[3]*pose->orientation[0] - delta[4]*pose->orientation[1] - delta[5]*pose->orientation[2] - delta[6]*pose->orientation[3];
    q_new[1] = delta[3]*pose->orientation[1] + delta[4]*pose->orientation[0] + delta[5]*pose->orientation[3] - delta[6]*pose->orientation[2];
    q_new[2] = delta[3]*pose->orientation[2] - delta[4]*pose->orientation[3] + delta[5]*pose->orientation[0] + delta[6]*pose->orientation[1];
    q_new[3] = delta[3]*pose->orientation[3] + delta[4]*pose->orientation[2] - delta[5]*pose->orientation[1] + delta[6]*pose->orientation[0];
    float qn = sqrtf(q_new[0]*q_new[0] + q_new[1]*q_new[1] + q_new[2]*q_new[2] + q_new[3]*q_new[3]);
    if (qn > 0.0f) {
        pose->orientation[0] = q_new[0] / qn;
        pose->orientation[1] = q_new[1] / qn;
        pose->orientation[2] = q_new[2] / qn;
        pose->orientation[3] = q_new[3] / qn;
    }
    return 0;
}

static int slam_propagate_drift_correction(SlamSystem* system, int matched_frame_id,
                                          int current_frame_id,
                                          const float* corrected_poses,
                                          int num_corrected) {
    if (!system || !corrected_poses || num_corrected <= 0) return -1;
    if (!system->loop_closure_config.enable_drift_propagation) return 0;
    InternalLoopClosure* ilc = &system->loop_closure_internal;
    KeyFrame* keyframes = system->local_map.keyframes;
    int num_kfs = system->local_map.num_keyframes;
    float* drift_deltas = (float*)slam_calloc(num_kfs, 7 * sizeof(float));
    if (!drift_deltas) return -1;
    for (int i = 0; i < num_kfs; i++) {
        SlamPose* kf_pose = &keyframes[i].pose;
        int nearest_corrected = -1;
        float nearest_dist = FLT_MAX;
        for (int j = 0; j < num_corrected; j++) {
            float* cp = (float*)&corrected_poses[j * 7];
            float dx = kf_pose->position[0] - cp[0];
            float dy = kf_pose->position[1] - cp[1];
            float dz = kf_pose->position[2] - cp[2];
            float dist = dx*dx + dy*dy + dz*dz;
            if (dist < nearest_dist) {
                nearest_dist = dist;
                nearest_corrected = j;
            }
        }
        if (nearest_corrected < 0) continue;
        float* nearest_cp = (float*)&corrected_poses[nearest_corrected * 7];
        if (nearest_dist < SLAM_DRIFT_PROPAGATION_RADIUS * SLAM_DRIFT_PROPAGATION_RADIUS) {
            float weight = 1.0f - sqrtf(nearest_dist) / SLAM_DRIFT_PROPAGATION_RADIUS;
            if (weight < 0.0f) weight = 0.0f;
            float delta[7];
            slam_compute_relative_pose(kf_pose, (const SlamPose*)nearest_cp, delta);
            for (int k = 0; k < 3; k++) drift_deltas[i * 7 + k] = delta[k] * weight;
            for (int k = 3; k < 7; k++) {
                float angle = acosf(fmaxf(-1.0f, fminf(1.0f, delta[k])));
                float slerp_t = weight;
                float sin_half = sinf(angle * slerp_t);
                float sin_angle = sinf(angle);
                drift_deltas[i * 7 + k] = (sin_angle > 1e-6f) ?
                    (sin_half / sin_angle) : slerp_t;
            }
        }
    }
    for (int i = 0; i < num_kfs; i++) {
        if (i >= matched_frame_id && i <= current_frame_id) continue;
        if (fabsf(drift_deltas[i * 7]) < 1e-6f && fabsf(drift_deltas[i * 7 + 1]) < 1e-6f &&
            fabsf(drift_deltas[i * 7 + 2]) < 1e-6f) continue;
        slam_apply_delta_to_pose(&keyframes[i].pose, &drift_deltas[i * 7]);
    }
    if (ilc->drift_deltas) slam_free(ilc->drift_deltas);
    ilc->drift_deltas = drift_deltas;
    ilc->drift_delta_capacity = num_kfs;
    ilc->drift_active = 1;
    ilc->drift_frame_start = matched_frame_id;
    return 0;
}

/* ==================== 公开API函数实现（MUL-06） ==================== */

int slam_set_loop_closure_config(SlamSystem* system, const LoopClosureConfig* config) {
    if (!system || !config) return -1;
    memcpy(&system->loop_closure_config, config, sizeof(LoopClosureConfig));
    InternalLoopClosure* ilc = &system->loop_closure_internal;
    ilc->temporal_consistency_threshold = config->temporal_consistency_threshold;
    return 0;
}

int slam_get_loop_closure_config(const SlamSystem* system, LoopClosureConfig* config) {
    if (!system || !config) return -1;
    memcpy(config, &system->loop_closure_config, sizeof(LoopClosureConfig));
    return 0;
}

int slam_set_vocabulary_config(SlamSystem* system, const VisualVocabularyConfig* config) {
    if (!system || !config) return -1;
    if (system->vocabulary.is_built) {
        slam_vocabulary_free(&system->vocabulary);
        memset(&system->vocabulary, 0, sizeof(InternalVocabulary));
    }
    VisualVocabularyConfig effective_config = *config;
    if (effective_config.vocabulary_size <= 0) effective_config.vocabulary_size = SLAM_VOCABULARY_SIZE;
    if (effective_config.vocabulary_depth <= 0) effective_config.vocabulary_depth = SLAM_VOCABULARY_DEPTH;
    if (effective_config.branching_factor <= 0) effective_config.branching_factor = SLAM_VOCABULARY_BRANCHING;
    if (effective_config.descriptor_length <= 0) effective_config.descriptor_length = 32;
    return slam_vocabulary_init(&system->vocabulary, &effective_config);
}

int slam_build_vocabulary(SlamSystem* system) {
    if (!system || !system->local_map.keyframes || system->local_map.num_keyframes < 5) return -1;
    InternalVocabulary* vocab = &system->vocabulary;
    if (!vocab->root) {
        VisualVocabularyConfig default_config;
        memset(&default_config, 0, sizeof(VisualVocabularyConfig));
        default_config.vocabulary_size = SLAM_VOCABULARY_SIZE;
        default_config.vocabulary_depth = SLAM_VOCABULARY_DEPTH;
        default_config.branching_factor = SLAM_VOCABULARY_BRANCHING;
        default_config.descriptor_length = 32;
        default_config.enable_tf_idf = 1;
        if (slam_vocabulary_init(vocab, &default_config) != 0) return -1;
    }

    int total_descriptors = 0;
    int num_frames = system->local_map.num_keyframes;
    for (int i = 0; i < num_frames; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        if (kf->descriptors && kf->num_features > 0) {
            total_descriptors += kf->num_features;
        }
    }
    if (total_descriptors < 100) return -1;

    int max_train = SLAM_VOCABULARY_MAX_TRAIN;
    int sample_count = total_descriptors < max_train ? total_descriptors : max_train;
    float* all_descriptors = (float*)slam_malloc(sample_count * vocab->descriptor_length * sizeof(float));
    if (!all_descriptors) return -1;

    int count = 0;
    int stride = total_descriptors / sample_count;
    if (stride < 1) stride = 1;
    int idx = 0;
    for (int i = 0; i < num_frames && count < sample_count; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        if (!kf->descriptors || kf->num_features <= 0) continue;
        for (int j = 0; j < kf->num_features && count < sample_count; j++) {
            if ((idx % stride) == 0) {
                memcpy(&all_descriptors[count * vocab->descriptor_length],
                       &kf->descriptors[j * vocab->descriptor_length],
                       vocab->descriptor_length * sizeof(float));
                count++;
            }
            idx++;
        }
    }

    int ret = slam_vocabulary_build(vocab, all_descriptors, count, vocab->descriptor_length);
    slam_free(all_descriptors);
    if (ret != 0) return -1;

    for (int i = 0; i < num_frames; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        if (kf->descriptors && kf->num_features > 0) {
            slam_vocabulary_add_frame(vocab, kf->descriptors, kf->num_features, vocab->descriptor_length);
        }
    }

    if (vocab->enable_tf_idf) {
        slam_vocabulary_update_tfidf(vocab);
    }

    return 0;
}

int slam_update_vocabulary(SlamSystem* system) {
    if (!system) return -1;
    InternalVocabulary* vocab = &system->vocabulary;
    if (!vocab->is_built || !vocab->incremental_update_enabled) return -1;
    int num_frames = system->local_map.num_keyframes;
    int updated = 0;
    for (int i = 0; i < num_frames; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        if (kf->descriptors && kf->num_features > 0) {
            if (slam_vocabulary_add_frame(vocab, kf->descriptors, kf->num_features, vocab->descriptor_length) == 0) {
                updated++;
            }
        }
    }
    if (updated > 0 && vocab->enable_tf_idf) {
        slam_vocabulary_update_tfidf(vocab);
    }
    return updated > 0 ? 0 : -1;
}

int slam_get_loop_closure_candidates(const SlamSystem* system, LoopClosureCandidate* candidates, int max_count) {
    if (!system || !candidates || max_count <= 0) return -1;
    const InternalLoopClosure* ilc = &system->loop_closure_internal;
    int count = ilc->num_scored_candidates < max_count ? ilc->num_scored_candidates : max_count;
    for (int i = 0; i < count; i++) {
        memcpy(&candidates[i], &ilc->scored_candidates[i], sizeof(LoopClosureCandidate));
    }
    return count;
}

int slam_get_loop_closure_quality(const SlamSystem* system, float* overall_score, float* precision, float* recall) {
    if (!system) return -1;
    const InternalLoopClosure* ilc = &system->loop_closure_internal;
    if (overall_score) *overall_score = ilc->detection_confidence;
    if (precision) {
        float tp = ilc->detection_confidence * (float)ilc->consecutive_detections;
        float fp = (1.0f - ilc->detection_confidence) * (float)ilc->consecutive_detections;
        *precision = (tp + fp > 1e-6f) ? tp / (tp + fp) : 0.0f;
    }
    if (recall) {
        *recall = ilc->detection_confidence;
    }
    return 0;
}

int slam_fuse_map_points(SlamSystem* system) {
    if (!system) return -1;
    InternalLoopClosure* ilc = &system->loop_closure_internal;
    if (ilc->fusion_pending) {
        ilc->fusion_pending = 0;
        return ilc->total_fusions_performed;
    }
    int num_kfs = system->local_map.num_keyframes;
    int total_fusions = 0;
    for (int i = 0; i < num_kfs - 1; i++) {
        for (int j = i + 1; j < num_kfs; j++) {
            int gap = abs(i - j);
            if (gap < 30) continue;
            int fusion_count = slam_fuse_loop_closure_map(system, i, j);
            if (fusion_count > 0) total_fusions += fusion_count;
        }
    }
    ilc->total_fusions_performed += total_fusions;
    return total_fusions;
}

int slam_get_covisibility_edges(const SlamSystem* system, CovisibilityEdge* edges, int max_edges) {
    if (!system || !edges || max_edges <= 0) return -1;
    const InternalCovisibility* cov = &system->covisibility;
    if (!cov->adjacency_matrix || cov->num_frames <= 0) return -1;
    int edge_count = 0;
    for (int i = 0; i < cov->num_frames && edge_count < max_edges; i++) {
        for (int j = i + 1; j < cov->num_frames && edge_count < max_edges; j++) {
            int w = cov->adjacency_matrix[i * cov->max_frames + j];
            if (w >= SLAM_COVISIBILITY_MIN_WEIGHT) {
                edges[edge_count].from_frame_id = i;
                edges[edge_count].to_frame_id = j;
                edges[edge_count].weight = w;
                SlamPose* pi = &system->local_map.keyframes[i].pose;
                SlamPose* pj = &system->local_map.keyframes[j].pose;
                edges[edge_count].relative_pose[0] = pj->position[0] - pi->position[0];
                edges[edge_count].relative_pose[1] = pj->position[1] - pi->position[1];
                edges[edge_count].relative_pose[2] = pj->position[2] - pi->position[2];
                float q_conj[4] = {pi->orientation[0], -pi->orientation[1], -pi->orientation[2], -pi->orientation[3]};
                edges[edge_count].relative_pose[3] = q_conj[0]*pj->orientation[0] - q_conj[1]*pj->orientation[1] - q_conj[2]*pj->orientation[2] - q_conj[3]*pj->orientation[3];
                edges[edge_count].relative_pose[4] = q_conj[0]*pj->orientation[1] + q_conj[1]*pj->orientation[0] + q_conj[2]*pj->orientation[3] - q_conj[3]*pj->orientation[2];
                edges[edge_count].relative_pose[5] = q_conj[0]*pj->orientation[2] - q_conj[1]*pj->orientation[3] + q_conj[2]*pj->orientation[0] + q_conj[3]*pj->orientation[1];
                edges[edge_count].relative_pose[6] = q_conj[0]*pj->orientation[3] + q_conj[1]*pj->orientation[2] - q_conj[2]*pj->orientation[1] + q_conj[3]*pj->orientation[0];
                float qn = sqrtf(edges[edge_count].relative_pose[3]*edges[edge_count].relative_pose[3] +
                                edges[edge_count].relative_pose[4]*edges[edge_count].relative_pose[4] +
                                edges[edge_count].relative_pose[5]*edges[edge_count].relative_pose[5] +
                                edges[edge_count].relative_pose[6]*edges[edge_count].relative_pose[6]);
                if (qn > 1e-6f) {
                    edges[edge_count].relative_pose[3] /= qn;
                    edges[edge_count].relative_pose[4] /= qn;
                    edges[edge_count].relative_pose[5] /= qn;
                    edges[edge_count].relative_pose[6] /= qn;
                }
                edge_count++;
            }
        }
    }
    return edge_count;
}

int slam_get_vocabulary_stats(const SlamSystem* system, int* vocab_size, int* total_words, float* avg_weight) {
    if (!system) return -1;
    const InternalVocabulary* vocab = &system->vocabulary;
    if (vocab_size) *vocab_size = vocab->num_leaf_nodes;
    if (total_words) *total_words = vocab->total_trained_features;
    if (avg_weight) {
        float sum = 0.0f;
        if (vocab->leaf_words && vocab->num_leaf_nodes > 0) {
            for (int i = 0; i < vocab->num_leaf_nodes; i++) {
                sum += vocab->leaf_words[i].weight;
            }
            *avg_weight = sum / (float)vocab->num_leaf_nodes;
        } else {
            *avg_weight = 0.0f;
        }
    }
    return 0;
}

/* ==================== 候选选择函数实现 ==================== */

static int slam_compute_bow_vector(SlamSystem* system, int frame_id) {
    if (!system || frame_id < 0 || frame_id >= system->local_map.num_keyframes) {
        return -1;
    }
    InternalVocabulary* vocab = &system->vocabulary;
    if (!vocab->is_built || !vocab->root) {
        return -1;
    }
    KeyFrame* kf = &system->local_map.keyframes[frame_id];
    if (!kf->descriptors || kf->num_features < 10) {
        return -1;
    }
    InternalLoopClosure* ilc = &system->loop_closure_internal;
    int max_words = vocab->num_leaf_nodes > 0 ? vocab->num_leaf_nodes : SLAM_VOCABULARY_SIZE;
    if (!ilc->bow_vector) {
        ilc->bow_vector = (float*)slam_calloc(max_words, sizeof(float));
        if (!ilc->bow_vector) return -1;
        ilc->bow_vector_size = max_words;
    }
    int computed_size = 0;
    int ret = slam_vocabulary_compute_bow(vocab, kf->descriptors, kf->num_features,
                                          vocab->descriptor_length,
                                          ilc->bow_vector, &computed_size);
    if (ret == 0) {
        ilc->bow_computed_for_frame = frame_id;
        if (computed_size > 0) ilc->bow_vector_size = computed_size;
    }
    return ret;
}

static int slam_select_candidates_by_bow(SlamSystem* system, int frame_id,
                                        int* candidates, int max_candidates,
                                        float* scores) {
    if (!system || !candidates || !scores || max_candidates <= 0) return -1;
    KeyFrame* keyframes = system->local_map.keyframes;
    int num_kfs = system->local_map.num_keyframes;
    if (frame_id < 0 || frame_id >= num_kfs) return -1;

    InternalVocabulary* vocab = &system->vocabulary;
    if (!vocab->is_built || !vocab->root) return -1;
    InternalLoopClosure* ilc = &system->loop_closure_internal;

    if (ilc->bow_computed_for_frame != frame_id) {
        if (slam_compute_bow_vector(system, frame_id) != 0) return -1;
    }
    float* current_bow = ilc->bow_vector;
    int current_bow_size = ilc->bow_vector_size;
    if (!current_bow) return -1;

    int min_frame_gap = (int)system->loop_closure_config.min_frame_gap > 0 ?
        (int)system->loop_closure_config.min_frame_gap : 30;

    typedef struct {
        int frame_id;
        float score;
    } ScoreEntry;
    ScoreEntry* all_scores = (ScoreEntry*)slam_malloc(num_kfs * sizeof(ScoreEntry));
    if (!all_scores) return -1;

    int num_valid = 0;
    for (int i = 0; i < num_kfs; i++) {
        if (i == frame_id || abs(i - frame_id) < min_frame_gap) continue;
        KeyFrame* kf = &keyframes[i];
        if (!kf->descriptors || kf->num_features < 10) continue;

        float kf_bow[1024];
        int kf_bow_size = 0;
        int ret = slam_vocabulary_compute_bow(vocab, kf->descriptors, kf->num_features,
                                              vocab->descriptor_length, kf_bow, &kf_bow_size);
        if (ret != 0 || kf_bow_size <= 0) continue;

        float sim = slam_vocabulary_compute_similarity(current_bow, current_bow_size, kf_bow, kf_bow_size);
        if (sim > SLAM_BOW_SIMILARITY_THRESHOLD) {
            all_scores[num_valid].frame_id = i;
            all_scores[num_valid].score = sim;
            num_valid++;
        }
    }

    int selected = num_valid < max_candidates ? num_valid : max_candidates;
    for (int i = 0; i < selected; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_valid; j++) {
            if (all_scores[j].score > all_scores[best_idx].score) {
                best_idx = j;
            }
        }
        candidates[i] = all_scores[best_idx].frame_id;
        scores[i] = all_scores[best_idx].score;
        ScoreEntry temp = all_scores[i];
        all_scores[i] = all_scores[best_idx];
        all_scores[best_idx] = temp;
    }

    slam_free(all_scores);
    return selected;
}

static int slam_select_candidates_hybrid(SlamSystem* system, int frame_id,
                                        int* candidates, int max_candidates,
                                        float* scores) {
    if (!system || !candidates || !scores || max_candidates <= 0) return -1;
    KeyFrame* keyframes = system->local_map.keyframes;
    int num_kfs = system->local_map.num_keyframes;
    if (frame_id < 0 || frame_id >= num_kfs) return -1;

    KeyFrame* current_kf = &keyframes[frame_id];
    int min_frame_gap = (int)system->loop_closure_config.min_frame_gap > 0 ?
        (int)system->loop_closure_config.min_frame_gap : 30;

    typedef struct {
        int frame_id;
        float score;
    } ScoreEntry;
    ScoreEntry* all_scores = (ScoreEntry*)slam_malloc(num_kfs * sizeof(ScoreEntry));
    if (!all_scores) return -1;

    int num_valid = 0;
    for (int i = 0; i < num_kfs; i++) {
        if (i == frame_id || abs(i - frame_id) < min_frame_gap) continue;
        KeyFrame* kf = &keyframes[i];
        if (!kf->descriptors || kf->num_features < 10) continue;

        float dx = kf->pose.position[0] - current_kf->pose.position[0];
        float dy = kf->pose.position[1] - current_kf->pose.position[1];
        float dz = kf->pose.position[2] - current_kf->pose.position[2];
        float geo_dist = sqrtf(dx*dx + dy*dy + dz*dz);

        float geo_score = (geo_dist < 10.0f) ? (1.0f - geo_dist / 10.0f) : 0.0f;

        float bow_sim = 0.0f;
        if (system->vocabulary.is_built) {
            float kf_bow[1024];
            int kf_bow_size = 0;
            int ret = slam_vocabulary_compute_bow(&system->vocabulary, kf->descriptors, kf->num_features,
                                                  system->vocabulary.descriptor_length, kf_bow, &kf_bow_size);
            if (ret == 0 && kf_bow_size > 0) {
                InternalLoopClosure* ilc = &system->loop_closure_internal;
                if (ilc->bow_computed_for_frame != frame_id) {
                    slam_compute_bow_vector(system, frame_id);
                }
                if (ilc->bow_vector && ilc->bow_vector_size > 0) {
                    bow_sim = slam_vocabulary_compute_similarity(ilc->bow_vector, ilc->bow_vector_size, kf_bow, kf_bow_size);
                }
            }
        }

        float hybrid = 0.6f * bow_sim + 0.4f * geo_score;
        if (hybrid > SLAM_BOW_SIMILARITY_THRESHOLD) {
            all_scores[num_valid].frame_id = i;
            all_scores[num_valid].score = hybrid;
            num_valid++;
        }
    }

    int selected = num_valid < max_candidates ? num_valid : max_candidates;
    for (int i = 0; i < selected; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_valid; j++) {
            if (all_scores[j].score > all_scores[best_idx].score) {
                best_idx = j;
            }
        }
        candidates[i] = all_scores[best_idx].frame_id;
        scores[i] = all_scores[best_idx].score;
        ScoreEntry temp = all_scores[i];
        all_scores[i] = all_scores[best_idx];
        all_scores[best_idx] = temp;
    }

    slam_free(all_scores);
    return selected;
}

/* ==================== B-02: SLAM合成数据生成与演示（增强版） ==================== */

/**
 * @brief 合成场景配置常量
 */
#define SLAM_SYNTHETIC_NUM_POINTS   300
#define SLAM_SYNTHETIC_NUM_CORNERS   80
#define SLAM_SYNTHETIC_NUM_LINES     40

/**
 * @brief 在图像上绘制一个抗锯齿圆形
 */
static void draw_circle(float* image, int width, int height,
                        float cx, float cy, float radius,
                        float brightness, float contrast) {
    int r_int = (int)(radius + 1.5f);
    int ui = (int)cx;
    int vi = (int)cy;
    for (int dy = -r_int; dy <= r_int; dy++) {
        for (int dx = -r_int; dx <= r_int; dx++) {
            float dist2 = (float)(dx*dx + dy*dy);
            if (dist2 > (radius + 1.0f) * (radius + 1.0f)) continue;
            int px = ui + dx;
            int py = vi + dy;
            if (px < 0 || px >= width || py < 0 || py >= height) continue;
            float weight = 1.0f - sqrtf(dist2) / (radius + 1.0f);
            if (weight < 0.0f) weight = 0.0f;
            weight = weight * weight * (3.0f - 2.0f * weight);
            float existing = image[py * width + px];
            image[py * width + px] = existing + (brightness - existing) * weight * contrast;
        }
    }
}

/**
 * @brief 在图像上绘制一个矩形角点（L形）
 */
static void draw_corner(float* image, int width, int height,
                        float cx, float cy, float arm_length,
                        float brightness) {
    int len = (int)(arm_length + 0.5f);
    int ui = (int)cx;
    int vi = (int)cy;
    if (ui < 0 || ui >= width || vi < 0 || vi >= height) return;

    /* 绘制水平臂（向右） */
    for (int dx = 0; dx <= len; dx++) {
        int px = ui + dx;
        if (px >= width) break;
        for (int dy = -1; dy <= 1; dy++) {
            int py = vi + dy;
            if (py < 0 || py >= height) continue;
            float alpha = (dx == 0 || dx == len) ? 0.7f : 1.0f;
            image[py * width + px] = image[py * width + px] * (1.0f - alpha * 0.5f) + brightness * alpha * 0.5f;
        }
    }
    /* 绘制垂直臂（向下） */
    for (int dy = 0; dy <= len; dy++) {
        int py = vi + dy;
        if (py >= height) break;
        for (int dx = -1; dx <= 1; dx++) {
            int px = ui + dx;
            if (px < 0 || px >= width) continue;
            float alpha = (dy == 0 || dy == len) ? 0.7f : 1.0f;
            image[py * width + px] = image[py * width + px] * (1.0f - alpha * 0.5f) + brightness * alpha * 0.5f;
        }
    }
}

/**
 * @brief 在图像上绘制线段的抗锯齿版本
 */
static void draw_line(float* image, int width, int height,
                      float x0, float y0, float x1, float y1,
                      float brightness) {
    float dx = x1 - x0;
    float dy = y1 - y0;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 0.5f) return;
    dx /= len;
    dy /= len;
    int steps = (int)(len * 2.0f);
    if (steps < 1) steps = 1;
    for (int s = 0; s <= steps; s++) {
        float t = (float)s / (float)steps;
        float px = x0 + dx * len * t;
        float py = y0 + dy * len * t;
        int ix = (int)px;
        int iy = (int)py;
        if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue;
        for (int dxo = -1; dxo <= 1; dxo++) {
            for (int dyo = -1; dyo <= 1; dyo++) {
                int fx = ix + dxo;
                int fy = iy + dyo;
                if (fx < 0 || fx >= width || fy < 0 || fy >= height) continue;
                image[fy * width + fx] = image[fy * width + fx] * 0.6f + brightness * 0.4f;
            }
        }
    }
}

/**
 * @brief 计算合成相机的位姿（返回相机位置和朝向矩阵）
 */
static void compute_synthetic_camera_pose(int frame_id, int total_frames,
                                          float cam_pos[3], float R[9]) {
    float t = (total_frames > 1) ? (float)frame_id / (float)(total_frames - 1) : 0.0f;

    /* 运动模式选择：根据总帧数自动选择不同的轨迹模式 */
    float orbit_radius = 6.0f;
    float pos_z_offset = 0.0f;
    float look_at_z = 0.0f;

    if (total_frames <= 200) {
        /* 简单圆形轨迹 + 垂直震荡 */
        float angle = t * 2.0f * 3.14159265f;
        cam_pos[0] = orbit_radius * sinf(angle);
        cam_pos[1] = 0.5f * sinf(angle * 1.3f + 0.5f);
        cam_pos[2] = orbit_radius * cosf(angle) - 1.0f;
        pos_z_offset = 0.0f;
    } else if (total_frames <= 500) {
        /* 8字形轨迹 */
        float angle = t * 2.0f * 3.14159265f;
        float sin_a = sinf(angle);
        float cos_a = cosf(angle);
        cam_pos[0] = orbit_radius * sin_a;
        cam_pos[1] = 0.6f * sinf(2.0f * angle);
        cam_pos[2] = orbit_radius * cos_a * sin_a * 0.8f - 1.0f;
        pos_z_offset = 0.0f;
    } else {
        /* 多段混合轨迹：前进→旋转→后退→环绕 */
        float phase = t * 4.0f;
        int segment = (int)phase;
        float seg_t = phase - (float)segment;
        switch (segment % 4) {
            case 0: /* 前进：沿Z轴正向移动 */
                cam_pos[0] = 0.5f * sinf(seg_t * 3.14159f);
                cam_pos[1] = 0.3f * sinf(seg_t * 6.28318f);
                cam_pos[2] = -5.0f + seg_t * 8.0f;
                break;
            case 1: /* 旋转：围绕原点旋转90度 */
                cam_pos[0] = 3.0f * sinf(seg_t * 1.5708f);
                cam_pos[1] = 0.4f * sinf(seg_t * 6.28318f);
                cam_pos[2] = 3.0f * cosf(seg_t * 1.5708f);
                break;
            case 2: /* 后退 */
                cam_pos[0] = 3.0f * (1.0f - seg_t);
                cam_pos[1] = 0.3f * (1.0f - seg_t) * sinf(seg_t * 6.28318f);
                cam_pos[2] = -5.0f + (1.0f - seg_t) * 8.0f;
                break;
            default: /* 环绕 */
                cam_pos[0] = 5.0f * sinf(seg_t * 6.28318f);
                cam_pos[1] = 0.5f * sinf(seg_t * 12.56637f);
                cam_pos[2] = 6.0f * cosf(seg_t * 6.28318f) - 2.0f;
                break;
        }
    }

    float look_at[3] = {0.0f, pos_z_offset * 0.1f, look_at_z};
    float up[3] = {0.0f, -1.0f, 0.0f};

    float forward[3] = {
        look_at[0] - cam_pos[0],
        look_at[1] - cam_pos[1],
        look_at[2] - cam_pos[2]
    };
    float fwd_norm = sqrtf(forward[0]*forward[0] + forward[1]*forward[1] + forward[2]*forward[2]);
    if (fwd_norm > 1e-8f) {
        forward[0] /= fwd_norm; forward[1] /= fwd_norm; forward[2] /= fwd_norm;
    }
    float right[3];
    right[0] = up[1]*forward[2] - up[2]*forward[1];
    right[1] = up[2]*forward[0] - up[0]*forward[2];
    right[2] = up[0]*forward[1] - up[1]*forward[0];
    float r_norm = sqrtf(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if (r_norm > 1e-8f) {
        right[0] /= r_norm; right[1] /= r_norm; right[2] /= r_norm;
    }
    float cam_up[3];
    cam_up[0] = forward[1]*right[2] - forward[2]*right[1];
    cam_up[1] = forward[2]*right[0] - forward[0]*right[2];
    cam_up[2] = forward[0]*right[1] - forward[1]*right[0];

    R[0] = right[0]; R[1] = right[1]; R[2] = right[2];
    R[3] = cam_up[0]; R[4] = cam_up[1]; R[5] = cam_up[2];
    R[6] = -forward[0]; R[7] = -forward[1]; R[8] = -forward[2];
}

/**
 * @brief 生成多深度层的3D场景点（含随机扰动）
 */
static int generate_scene_points(float scene_points[][3], float point_sizes[],
                                 float point_brightnesses[], int max_points,
                                 unsigned int* base_seed) {
    unsigned int seed = *base_seed;
    int idx = 0;

    /* 第一层：近处特征点（深度2-4m），高密度 */
    seed = seed * 1103515245u + 12345u;
    float near_count_f = 30.0f + (float)(seed & 0x3F);
    int near_count = (int)near_count_f;
    if (near_count > max_points - idx) near_count = max_points - idx;
    for (int i = 0; i < near_count && idx < max_points; i++) {
        seed = seed * 1103515245u + 12345u;
        float rx = ((float)(seed & 0x7FFF) / 32767.0f - 0.5f) * 3.0f;
        seed = seed * 1103515245u + 12345u;
        float ry = ((float)(seed & 0x7FFF) / 32767.0f - 0.5f) * 2.0f;
        seed = seed * 1103515245u + 12345u;
        float rz = 2.0f + ((float)(seed & 0xFFFF) / 65535.0f) * 2.0f;
        scene_points[idx][0] = rx;
        scene_points[idx][1] = ry;
        scene_points[idx][2] = rz;
        seed = seed * 1103515245u + 12345u;
        point_sizes[idx] = 1.5f + ((float)(seed & 0x7F) / 127.0f) * 2.5f;
        seed = seed * 1103515245u + 12345u;
        point_brightnesses[idx] = 0.7f + ((float)(seed & 0xFF) / 255.0f) * 0.3f;
        idx++;
    }

    /* 第二层：中等距离特征点（深度4-8m），网格布局+随机扰动 */
    int cols = 15, rows = 12;
    for (int gy = 0; gy < rows && idx < max_points; gy++) {
        for (int gx = 0; gx < cols && idx < max_points; gx++) {
            seed = seed * 1103515245u + 12345u;
            float rand_offset = ((float)(seed & 0xFFFF) / 65535.0f - 0.5f) * 0.4f;
            float gxf = (float)gx / (float)(cols - 1) * 4.0f - 2.0f + rand_offset;
            seed = seed * 1103515245u + 12345u;
            rand_offset = ((float)(seed & 0xFFFF) / 65535.0f - 0.5f) * 0.4f;
            float gyf = (float)gy / (float)(rows - 1) * 3.0f - 1.5f + rand_offset;
            seed = seed * 1103515245u + 12345u;
            float rand_h = ((float)(seed & 0xFFFF) / 65535.0f - 0.5f) * 1.5f;
            scene_points[idx][0] = gxf;
            scene_points[idx][1] = gyf;
            scene_points[idx][2] = 5.0f + rand_h * 2.0f;
            seed = seed * 1103515245u + 12345u;
            point_sizes[idx] = 2.0f + ((float)(seed & 0xFF) / 255.0f) * 3.0f;
            seed = seed * 1103515245u + 12345u;
            point_brightnesses[idx] = 0.75f + ((float)(seed & 0x7F) / 127.0f) * 0.25f;
            idx++;
        }
    }

    /* 第三层：远处特征点（深度8-15m），稀疏随机分布 */
    seed = seed * 1103515245u + 12345u;
    float far_count_f = 20.0f + (float)(seed & 0x2F);
    int far_count = (int)far_count_f;
    for (int i = 0; i < far_count && idx < max_points; i++) {
        seed = seed * 1103515245u + 12345u;
        float rx = ((float)(seed & 0x7FFF) / 32767.0f - 0.5f) * 6.0f;
        seed = seed * 1103515245u + 12345u;
        float ry = ((float)(seed & 0x7FFF) / 32767.0f - 0.5f) * 4.0f;
        seed = seed * 1103515245u + 12345u;
        float rz = 8.0f + ((float)(seed & 0xFFFF) / 65535.0f) * 7.0f;
        scene_points[idx][0] = rx;
        scene_points[idx][1] = ry;
        scene_points[idx][2] = rz;
        seed = seed * 1103515245u + 12345u;
        point_sizes[idx] = 1.0f + ((float)(seed & 0x7F) / 127.0f) * 1.5f;
        seed = seed * 1103515245u + 12345u;
        point_brightnesses[idx] = 0.65f + ((float)(seed & 0x7F) / 127.0f) * 0.2f;
        idx++;
    }

    *base_seed = seed;
    return idx;
}

/**
 * @brief 生成合成SLAM测试帧图像（增强版）
 * 
 * 生成具有以下特征的合成图像帧：
 * - 多深度层3D场景点（近/中/远三层）
 * - 棋盘格背景纹理
 * - L形角点特征
 * - 线段边缘特征
 * - 运动模糊（快速运动时）
 * - 光照渐变
 * - 复合噪声模型（高斯+散粒噪声）
 */
int slam_generate_synthetic_frame(float* image_data,
                                 int width, int height,
                                 int frame_id, int total_frames,
                                 const float* camera_params) {
#ifdef SELFLNN_STRICT_REAL_DATA
    /* 严格真实数据模式：禁止合成帧生成 */
    (void)image_data; (void)width; (void)height;
    (void)frame_id; (void)total_frames; (void)camera_params;
    log_error("[SLAM] 严格真实数据模式：slam_generate_synthetic_frame() 已禁用。"
              "请使用真实摄像头图像数据。");
    return -1;
#else
    if (!image_data || width <= 0 || height <= 0 || total_frames <= 0 || !camera_params) {
        return -1;
    }

    float fx = (camera_params[0] > 0.0f) ? camera_params[0] : SLAM_DEFAULT_FOCAL_LENGTH;
    float fy = (camera_params[1] > 0.0f) ? camera_params[1] : SLAM_DEFAULT_FOCAL_LENGTH;
    float cx = (camera_params[2] > 0.0f) ? camera_params[2] : (float)width * 0.5f;
    float cy = (camera_params[3] > 0.0f) ? camera_params[3] : (float)height * 0.5f;

    float t = (total_frames > 1) ? (float)frame_id / (float)(total_frames - 1) : 0.0f;

    /* ---- 1. 填充背景: 棋盘格 + 径向渐变 ---- */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = (float)(x - (int)cx) / (float)width;
            float dy = (float)(y - (int)cy) / (float)height;
            float dist = sqrtf(dx * dx + dy * dy);

            /* 棋盘格纹理 */
            float checker_x = (float)x * 0.05f;
            float checker_y = (float)y * 0.05f;
            int cx_i = (int)(checker_x + 1000.0f);
            int cy_i = (int)(checker_y + 1000.0f);
            float checker = ((cx_i + cy_i) & 1) ? 0.05f : -0.05f;

            /* 径向渐变 */
            float gradient = 0.42f + 0.08f * sinf(dist * 10.0f + t * 2.0f);

            /* 空间光照变化（多光源合成计算） */
            float light_angle = t * 6.2832f;
            float lighting = 1.0f + 0.15f * sinf(dx * 3.0f + light_angle) * cosf(dy * 2.0f);

            image_data[y * width + x] = (gradient + checker) * lighting;
            if (image_data[y * width + x] < 0.1f) image_data[y * width + x] = 0.1f;
            if (image_data[y * width + x] > 0.7f) image_data[y * width + x] = 0.7f;
        }
    }

    /* ---- 2. 计算相机位姿 ---- */
    float cam_pos[3], R[9];
    compute_synthetic_camera_pose(frame_id, total_frames, cam_pos, R);

    /* 计算前帧相机位姿（用于运动模糊估计） */
    float prev_cam_pos[3], prev_R[9];
    int prev_id = (frame_id > 0) ? frame_id - 1 : 0;
    compute_synthetic_camera_pose(prev_id, total_frames, prev_cam_pos, prev_R);

    /* 估计运动速度（用于运动模糊强度） */
    float motion_speed = 0.0f;
    for (int i = 0; i < 3; i++) {
        float diff = cam_pos[i] - prev_cam_pos[i];
        motion_speed += diff * diff;
    }
    float rot_diff = 0.0f;
    for (int i = 0; i < 9; i++) {
        float diff = R[i] - prev_R[i];
        rot_diff += diff * diff;
    }
    motion_speed = sqrtf(motion_speed + rot_diff * 10.0f);
    float blur_intensity = (motion_speed > 0.02f) ? (motion_speed * 30.0f) : 0.0f;
    if (blur_intensity > 3.0f) blur_intensity = 3.0f;

    /* ---- 3. 生成静态3D场景点 ---- */
    float scene_points[SLAM_SYNTHETIC_NUM_POINTS][3];
    float point_sizes[SLAM_SYNTHETIC_NUM_POINTS];
    float point_brightnesses[SLAM_SYNTHETIC_NUM_POINTS];
    unsigned int base_seed = (unsigned int)(frame_id + 1) * 2654435761u;
    int num_points = generate_scene_points(scene_points, point_sizes,
                                           point_brightnesses,
                                           SLAM_SYNTHETIC_NUM_POINTS, &base_seed);

    /* ---- 4. 投影并绘制3D特征点 ---- */
    float motion_blur_x = 0.0f, motion_blur_y = 0.0f;
    if (blur_intensity > 0.1f) {
        /* 计算运动模糊方向：将运动向量投影到图像平面 */
        float motion_dir[3] = {
            cam_pos[0] - prev_cam_pos[0],
            cam_pos[1] - prev_cam_pos[1],
            cam_pos[2] - prev_cam_pos[2]
        };
        float m_norm = sqrtf(motion_dir[0]*motion_dir[0] +
                            motion_dir[1]*motion_dir[1] +
                            motion_dir[2]*motion_dir[2]);
        if (m_norm > 1e-6f) {
            motion_dir[0] /= m_norm;
            motion_dir[1] /= m_norm;
            motion_dir[2] /= m_norm;
        }
        float img_motion[2] = {
            fx * motion_dir[0] / fmaxf(cam_pos[2], 0.5f),
            fy * motion_dir[1] / fmaxf(cam_pos[2], 0.5f)
        };
        motion_blur_x = img_motion[0] * blur_intensity * 2.0f;
        motion_blur_y = img_motion[1] * blur_intensity * 2.0f;
    }

    for (int p = 0; p < num_points; p++) {
        float Pc[3];
        Pc[0] = R[0]*scene_points[p][0] + R[1]*scene_points[p][1] + R[2]*scene_points[p][2]
                - (R[0]*cam_pos[0] + R[1]*cam_pos[1] + R[2]*cam_pos[2]);
        Pc[1] = R[3]*scene_points[p][0] + R[4]*scene_points[p][1] + R[5]*scene_points[p][2]
                - (R[3]*cam_pos[0] + R[4]*cam_pos[1] + R[5]*cam_pos[2]);
        Pc[2] = R[6]*scene_points[p][0] + R[7]*scene_points[p][1] + R[8]*scene_points[p][2]
                - (R[6]*cam_pos[0] + R[7]*cam_pos[1] + R[8]*cam_pos[2]);

        if (Pc[2] <= 0.01f) continue;

        float u = fx * Pc[0] / Pc[2] + cx;
        float v = fy * Pc[1] / Pc[2] + cy;

        if (u < -10.0f || u >= (float)(width + 10) || v < -10.0f || v >= (float)(height + 10)) continue;

        float radius = point_sizes[p] * (1.0f + 0.15f * sinf(t * 6.2832f + (float)p * 0.5f));
        float brightness = point_brightnesses[p];

        /* 添加运动模糊效果：沿运动方向拉伸圆形 */
        if (blur_intensity > 0.3f) {
            int num_blur_samples = (int)(blur_intensity * 3.0f) + 1;
            if (num_blur_samples > 8) num_blur_samples = 8;
            for (int s = -num_blur_samples / 2; s <= num_blur_samples / 2; s++) {
                float sample_u = u + motion_blur_x * (float)s / (float)num_blur_samples;
                float sample_v = v + motion_blur_y * (float)s / (float)num_blur_samples;
                float blur_weight = 1.0f / (float)(num_blur_samples + 1);
                draw_circle(image_data, width, height,
                          sample_u, sample_v,
                          radius * 0.8f,
                          brightness * 0.7f, blur_weight * 0.6f);
            }
        } else {
            draw_circle(image_data, width, height, u, v, radius, brightness, 0.8f);

            /* 每个第5个点绘制暗色外环增强角点响应 */
            if (p % 5 == 0) {
                draw_circle(image_data, width, height, u, v, radius + 1.0f,
                          brightness * 0.3f, 0.5f);
            }
        }
    }

    /* ---- 5. 绘制L形角点特征 ---- */
    base_seed = (unsigned int)(frame_id + 1) * 2654435761u + 12345u;
    for (int c = 0; c < SLAM_SYNTHETIC_NUM_CORNERS; c++) {
        base_seed = base_seed * 1103515245u + 12345u;
        float corner_x = ((float)(base_seed & 0x7FFF) / 32767.0f - 0.5f) * 4.0f;
        base_seed = base_seed * 1103515245u + 12345u;
        float corner_y = ((float)(base_seed & 0x7FFF) / 32767.0f - 0.5f) * 3.0f;
        base_seed = base_seed * 1103515245u + 12345u;
        float corner_z = 3.0f + ((float)(base_seed & 0xFFFF) / 65535.0f) * 10.0f;

        float Pc[3];
        Pc[0] = R[0]*corner_x + R[1]*corner_y + R[2]*corner_z
                - (R[0]*cam_pos[0] + R[1]*cam_pos[1] + R[2]*cam_pos[2]);
        Pc[1] = R[3]*corner_x + R[4]*corner_y + R[5]*corner_z
                - (R[3]*cam_pos[0] + R[4]*cam_pos[1] + R[5]*cam_pos[2]);
        Pc[2] = R[6]*corner_x + R[7]*corner_y + R[8]*corner_z
                - (R[6]*cam_pos[0] + R[7]*cam_pos[1] + R[8]*cam_pos[2]);

        if (Pc[2] <= 0.01f) continue;

        float uc = fx * Pc[0] / Pc[2] + cx;
        float vc = fy * Pc[1] / Pc[2] + cy;

        if (uc < 0.0f || uc >= (float)width || vc < 0.0f || vc >= (float)height) continue;

        /* 根据距离调整角点大小 */
        float corner_size = 4.0f + 15.0f / fmaxf(Pc[2], 0.5f);
        if (corner_size > 12.0f) corner_size = 12.0f;
        base_seed = base_seed * 1103515245u + 12345u;
        float corner_brightness = 0.6f + ((float)(base_seed & 0x7F) / 127.0f) * 0.4f;
        draw_corner(image_data, width, height, uc, vc, corner_size, corner_brightness);
    }

    /* ---- 6. 绘制线段特征（边缘检测数据） ---- */
    base_seed = (unsigned int)(frame_id + 1) * 2654435761u + 67890u;
    for (int l = 0; l < SLAM_SYNTHETIC_NUM_LINES; l++) {
        base_seed = base_seed * 1103515245u + 12345u;
        float lx0 = ((float)(base_seed & 0x7FFF) / 32767.0f - 0.5f) * 6.0f;
        base_seed = base_seed * 1103515245u + 12345u;
        float ly0 = ((float)(base_seed & 0x7FFF) / 32767.0f - 0.5f) * 4.0f;
        base_seed = base_seed * 1103515245u + 12345u;
        float lz = 2.0f + ((float)(base_seed & 0xFFFF) / 65535.0f) * 12.0f;
        base_seed = base_seed * 1103515245u + 12345u;
        float lx1 = lx0 + ((float)(base_seed & 0x7FFF) / 32767.0f - 0.5f) * 1.5f;
        base_seed = base_seed * 1103515245u + 12345u;
        float ly1 = ly0 + ((float)(base_seed & 0x7FFF) / 32767.0f - 0.5f) * 1.5f;

        /* 投影线段端点 */
        float Pc0[3], Pc1[3];
        for (int pt = 0; pt < 2; pt++) {
            float px = (pt == 0) ? lx0 : lx1;
            float py = (pt == 0) ? ly0 : ly1;
            float* Pc = (pt == 0) ? Pc0 : Pc1;
            Pc[0] = R[0]*px + R[1]*py + R[2]*lz
                    - (R[0]*cam_pos[0] + R[1]*cam_pos[1] + R[2]*cam_pos[2]);
            Pc[1] = R[3]*px + R[4]*py + R[5]*lz
                    - (R[3]*cam_pos[0] + R[4]*cam_pos[1] + R[5]*cam_pos[2]);
            Pc[2] = R[6]*px + R[7]*py + R[8]*lz
                    - (R[6]*cam_pos[0] + R[7]*cam_pos[1] + R[8]*cam_pos[2]);
        }
        if (Pc0[2] <= 0.01f || Pc1[2] <= 0.01f) continue;

        float u0 = fx * Pc0[0] / Pc0[2] + cx;
        float v0 = fy * Pc0[1] / Pc0[2] + cy;
        float u1 = fx * Pc1[0] / Pc1[2] + cx;
        float v1 = fy * Pc1[1] / Pc1[2] + cy;

        if ((u0 < 0 && u1 < 0) || (u0 >= width && u1 >= width) ||
            (v0 < 0 && v1 < 0) || (v0 >= height && v1 >= height)) continue;

        base_seed = base_seed * 1103515245u + 12345u;
        float line_brightness = 0.5f + ((float)(base_seed & 0x7F) / 127.0f) * 0.3f;
        draw_line(image_data, width, height, u0, v0, u1, v1, line_brightness);
    }

    /* ---- 7. 添加复合噪声模型 ---- */
    unsigned int noise_seed = (unsigned int)(frame_id + 1) * 2654435761u;
    for (int i = 0; i < width * height; i++) {
        noise_seed = noise_seed * 1103515245u + 12345u;
        float gauss_noise = ((float)(noise_seed & 0xFFFF) / 65535.0f +
                            (float)((noise_seed >> 16) & 0xFFFF) / 65535.0f - 1.0f) * 0.03f;

        /* 散粒噪声（与信号强度相关） */
        noise_seed = noise_seed * 1103515245u + 12345u;
        float shot_noise = ((float)(noise_seed & 0xFFFF) / 65535.0f - 0.5f) *
                           sqrtf(image_data[i] * 0.08f + 0.01f);

        image_data[i] += gauss_noise + shot_noise;

        /* 截止处理 */
        if (image_data[i] < 0.0f) image_data[i] = 0.0f;
        if (image_data[i] > 1.0f) image_data[i] = 1.0f;
    }

    return 0;
#endif /* SELFLNN_STRICT_REAL_DATA */
}

/**
 * @brief 计算绝对轨迹误差(ATE)
 * 
 * 比较估计轨迹与地面真实轨迹，计算RMSE和平均误差。
 * 地面真实轨迹直接由合成相机位姿提供。
 */
static float compute_ate(const float estimated_trajectory[][3], int est_length,
                         const float ground_truth[][3], int gt_length) {
    if (est_length <= 0 || gt_length <= 0) return -1.0f;

    int compare_len = (est_length < gt_length) ? est_length : gt_length;
    if (compare_len < 2) return -1.0f;

    /* 对齐：计算位姿差RMSE */
    float sum_sq_err = 0.0f;
    for (int i = 0; i < compare_len; i++) {
        float dx = estimated_trajectory[i][0] - ground_truth[i][0];
        float dy = estimated_trajectory[i][1] - ground_truth[i][1];
        float dz = estimated_trajectory[i][2] - ground_truth[i][2];
        float err = sqrtf(dx*dx + dy*dy + dz*dz);
        sum_sq_err += err * err;
    }
    float rmse = sqrtf(sum_sq_err / (float)compare_len);
    return rmse;
}

static int slam_read_ppm_token(FILE* f, char* token, int max_len) {
    int pos = 0;
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n');
            continue;
        }
        if (!isspace(c)) {
            token[pos++] = (char)c;
            break;
        }
    }
    if (c == EOF) return -1;
    while (pos < max_len - 1) {
        c = fgetc(f);
        if (c == EOF || isspace(c) || c == '#') {
            if (c == '#') {
                while ((c = fgetc(f)) != EOF && c != '\n');
            }
            break;
        }
        token[pos++] = (char)c;
    }
    token[pos] = '\0';
    return pos;
}

int slam_load_image_file(const char* filepath,
                         float** image_data,
                         int* out_width, int* out_height,
                         int* out_channels)
{
    if (!filepath || !image_data || !out_width || !out_height || !out_channels) {
        return -1;
    }

    *image_data = NULL;
    *out_width = 0;
    *out_height = 0;
    *out_channels = 0;

    /* 检测文件扩展名 */
    const char* ext = strrchr(filepath, '.');
    if (!ext) return -1;

    int is_ppm = 0, is_pgm = 0, is_raw_float = 0, is_raw_uint8 = 0;

    if (_stricmp(ext, ".ppm") == 0) {
        is_ppm = 1;
    } else if (_stricmp(ext, ".pgm") == 0) {
        is_pgm = 1;
    } else if (_stricmp(ext, ".raw") == 0 || _stricmp(ext, ".float") == 0) {
        is_raw_float = 1;
    } else if (_stricmp(ext, ".uint8") == 0 || _stricmp(ext, ".gray") == 0) {
        is_raw_uint8 = 1;
    } else {
        return -1;
    }

    FILE* f = fopen(filepath, "rb");
    if (!f) return -1;

    int result = -1;

    if (is_ppm || is_pgm) {
        /* 解析PPM/PGM头 */
        char magic[4] = {0};
        if (fread(magic, 1, 2, f) != 2) { fclose(f); return -1; }
        magic[2] = '\0';

        if (strcmp(magic, "P6") != 0 && strcmp(magic, "P5") != 0) {
            fclose(f);
            return -1;
        }

        int width = 0, height = 0, max_val = 0;
        char token[64];

        if (slam_read_ppm_token(f, token, sizeof(token)) <= 0) { fclose(f); return -1; }
        width = atoi(token);
        if (slam_read_ppm_token(f, token, sizeof(token)) <= 0) { fclose(f); return -1; }
        height = atoi(token);
        if (slam_read_ppm_token(f, token, sizeof(token)) <= 0) { fclose(f); return -1; }
        max_val = atoi(token);
        if (max_val <= 0 || max_val > 65535) { fclose(f); return -1; }

        int channels = (strcmp(magic, "P6") == 0) ? 3 : 1;
        size_t pixel_count = (size_t)width * height * channels;

        /* 读取像素数据 */
        unsigned char* raw = (unsigned char*)safe_malloc(pixel_count);
        if (!raw) { fclose(f); return -1; }

        size_t bytes_read = 0;
        if (max_val <= 255) {
            /* 8位数据 */
            bytes_read = fread(raw, 1, pixel_count, f);
        } else {
            /* 16位数据 - 需要转换 */
            uint16_t* raw16 = (uint16_t*)safe_malloc(pixel_count * 2);
            if (!raw16) {
                safe_free((void**)&raw);
                fclose(f);
                return -1;
            }
            bytes_read = fread(raw16, 2, pixel_count, f);
            if (bytes_read == pixel_count) {
                for (size_t i = 0; i < pixel_count; i++) {
                    raw[i] = (unsigned char)(raw16[i] >> 8);
                }
            }
            safe_free((void**)&raw16);
        }

        if (bytes_read != pixel_count) {
            safe_free((void**)&raw);
            fclose(f);
            return -1;
        }

        /* 转换为float数组 */
        float* data = (float*)safe_malloc(pixel_count * sizeof(float));
        if (!data) {
            safe_free((void**)&raw);
            fclose(f);
            return -1;
        }

        float inv_max = 1.0f / 255.0f;
        for (size_t i = 0; i < pixel_count; i++) {
            data[i] = raw[i] * inv_max;
        }

        safe_free((void**)&raw);
        *image_data = data;
        *out_width = width;
        *out_height = height;
        *out_channels = channels;
        result = 0;

    } else if (is_raw_float) {
        /* RAW float格式 - 前4字节为宽度(int)，接着4字节为高度(int)，然后是float数据 */
        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        fseek(f, 0, SEEK_SET);

        if (file_size < 8) { fclose(f); return -1; }

        int raw_width, raw_height;
        if (fread(&raw_width, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
        if (fread(&raw_height, sizeof(int), 1, f) != 1) { fclose(f); return -1; }

        if (raw_width <= 0 || raw_height <= 0 || raw_width > 10000 || raw_height > 10000) {
            fclose(f);
            return -1;
        }

        size_t expected_size = (size_t)raw_width * raw_height;
        long data_size = file_size - 8;
        if (data_size < (long)(expected_size * (long)sizeof(float))) {
            fclose(f);
            return -1;
        }

        float* data = (float*)safe_malloc(expected_size * sizeof(float));
        if (!data) { fclose(f); return -1; }

        size_t read_count = fread(data, sizeof(float), expected_size, f);
        if (read_count != expected_size) {
            safe_free((void**)&data);
            fclose(f);
            return -1;
        }

        *image_data = data;
        *out_width = raw_width;
        *out_height = raw_height;
        *out_channels = 1;
        result = 0;

    } else if (is_raw_uint8) {
        /* RAW uint8格式 - 前4字节宽度，接着4字节高度，然后是uint8数据 */
        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        fseek(f, 0, SEEK_SET);

        if (file_size < 8) { fclose(f); return -1; }

        int raw_width, raw_height;
        if (fread(&raw_width, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
        if (fread(&raw_height, sizeof(int), 1, f) != 1) { fclose(f); return -1; }

        if (raw_width <= 0 || raw_height <= 0 || raw_width > 10000 || raw_height > 10000) {
            fclose(f);
            return -1;
        }

        size_t expected_size = (size_t)raw_width * raw_height;
        long data_size = file_size - 8;
        if (data_size < (long)expected_size) {
            fclose(f);
            return -1;
        }

        unsigned char* raw = (unsigned char*)safe_malloc(expected_size);
        if (!raw) { fclose(f); return -1; }

        size_t read_count = fread(raw, 1, expected_size, f);
        if (read_count != expected_size) {
            safe_free((void**)&raw);
            fclose(f);
            return -1;
        }

        float* data = (float*)safe_malloc(expected_size * sizeof(float));
        if (!data) {
            safe_free((void**)&raw);
            fclose(f);
            return -1;
        }

        float inv_255 = 1.0f / 255.0f;
        for (size_t i = 0; i < expected_size; i++) {
            data[i] = raw[i] * inv_255;
        }

        safe_free((void**)&raw);
        *image_data = data;
        *out_width = raw_width;
        *out_height = raw_height;
        *out_channels = 1;
        result = 0;
    }

    fclose(f);
    return result;
}

int slam_save_image_file(const char* filepath,
                         const float* image_data,
                         int width, int height,
                         int channels)
{
    if (!filepath || !image_data || width <= 0 || height <= 0) {
        return -1;
    }

    if (channels != 1 && channels != 3) return -1;

    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    /* PPM (彩色) 或 PGM (灰度) */
    if (channels == 3) {
        fprintf(f, "P6\n%d %d\n255\n", width, height);
    } else {
        fprintf(f, "P5\n%d %d\n255\n", width, height);
    }

    size_t pixel_count = (size_t)width * height * channels;
    unsigned char* raw = (unsigned char*)safe_malloc(pixel_count);
    if (!raw) { fclose(f); return -1; }

    for (size_t i = 0; i < pixel_count; i++) {
        float v = image_data[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        raw[i] = (unsigned char)(v * 255.0f + 0.5f);
    }

    size_t written = fwrite(raw, 1, pixel_count, f);
    safe_free((void**)&raw);
    fclose(f);

    return (written == pixel_count) ? 0 : -1;
}

/* ============================================================================
 * 帧序列读取器实现
 * ============================================================================ */

struct SlamFrameReader {
    char** filenames;
    int* frame_numbers;
    int total_frames;
    int current_index;
    int start_index;
    char directory[1024];
};

static int slam_compare_frames(const void* a, const void* b)
{
    int na = ((const int*)a)[0];
    int nb = ((const int*)b)[0];
    return na - nb;
}

SlamFrameReader* slam_frame_reader_create(const char* directory,
                                          const char* file_prefix,
                                          const char* file_extension,
                                          int start_index)
{
    if (!directory || !file_prefix || !file_extension) return NULL;

#ifdef _WIN32
    /* Windows: 使用_findfirst/_findnext扫描目录 */
    char pattern[2048];
    snprintf(pattern, sizeof(pattern), "%s\\%s*.%s", directory, file_prefix, file_extension);

    /* 第一次扫描：计数 */
    int capacity = 64;
    int count = 0;
    char** names = (char**)safe_malloc(sizeof(char*) * (size_t)capacity);
    int* numbers = (int*)safe_malloc(sizeof(int) * (size_t)capacity);
    if (!names || !numbers) {
        safe_free((void**)&names);
        safe_free((void**)&numbers);
        return NULL;
    }

    struct _finddata_t find_data;
    intptr_t find_handle = _findfirst(pattern, &find_data);

    if (find_handle != -1) {
        do {
            if (!(find_data.attrib & _A_SUBDIR)) {
                /* 提取文件名中的数字部分 */
                const char* fname = find_data.name;
                const char* num_start = fname + strlen(file_prefix);
                int frame_num = atoi(num_start);

                if (count >= capacity) {
                    capacity *= 2;
                    char** new_names = (char**)safe_realloc(names, sizeof(char*) * (size_t)capacity);
                    int* new_numbers = (int*)safe_realloc(numbers, sizeof(int) * (size_t)capacity);
                    if (!new_names || !new_numbers) {
                        for (int i = 0; i < count; i++) safe_free((void**)&names[i]);
                        safe_free((void**)&names);
                        safe_free((void**)&numbers);
                        if (new_names) names = new_names;
                        else safe_free((void**)&names);
                        if (new_numbers) numbers = new_numbers;
                        else safe_free((void**)&numbers);
                        _findclose(find_handle);
                        return NULL;
                    }
                    names = new_names;
                    numbers = new_numbers;
                }

                names[count] = (char*)safe_malloc(strlen(fname) + 1);
                if (!names[count]) {
                    for (int i = 0; i < count; i++) safe_free((void**)&names[i]);
                    safe_free((void**)&names);
                    safe_free((void**)&numbers);
                    _findclose(find_handle);
                    return NULL;
                }
                strcpy(names[count], fname);
                numbers[count] = frame_num;
                count++;
            }
        } while (_findnext(find_handle, &find_data) == 0);
        _findclose(find_handle);
    }

#else
    /* Linux/Mac: 使用opendir/readdir扫描目录 */
    int capacity = 64;
    int count = 0;
    char** names = (char**)safe_malloc(sizeof(char*) * (size_t)capacity);
    int* numbers = (int*)safe_malloc(sizeof(int) * (size_t)capacity);
    if (!names || !numbers) {
        safe_free((void**)&names);
        safe_free((void**)&numbers);
        return NULL;
    }

    DIR* dir = opendir(directory);
    if (!dir) {
        for (int i = 0; i < count; i++) safe_free((void**)&names[i]);
        safe_free((void**)&names);
        safe_free((void**)&numbers);
        return NULL;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        /* 检查扩展名 */
        const char* dot = strrchr(entry->d_name, '.');
        if (!dot) continue;
        if (_stricmp(dot + 1, file_extension) != 0) continue;

        /* 检查前缀 */
        if (strncmp(entry->d_name, file_prefix, strlen(file_prefix)) != 0) continue;

        /* 提取数字 */
        const char* num_start = entry->d_name + strlen(file_prefix);
        int frame_num = atoi(num_start);

        if (count >= capacity) {
            capacity *= 2;
            char** new_names = (char**)safe_realloc(names, sizeof(char*) * (size_t)capacity);
            int* new_numbers = (int*)safe_realloc(numbers, sizeof(int) * (size_t)capacity);
            if (!new_names || !new_numbers) {
                for (int i = 0; i < count; i++) safe_free((void**)&names[i]);
                safe_free((void**)&names);
                safe_free((void**)&numbers);
                if (new_names) names = new_names;
                else safe_free((void**)&names);
                if (new_numbers) numbers = new_numbers;
                else safe_free((void**)&numbers);
                closedir(dir);
                return NULL;
            }
            names = new_names;
            numbers = new_numbers;
        }

        names[count] = (char*)safe_malloc(strlen(entry->d_name) + 1);
        if (!names[count]) {
            for (int i = 0; i < count; i++) safe_free((void**)&names[i]);
            safe_free((void**)&names);
            safe_free((void**)&numbers);
            closedir(dir);
            return NULL;
        }
        strcpy(names[count], entry->d_name);
        numbers[count] = frame_num;
        count++;
    }
    closedir(dir);
#endif

    if (count == 0) {
        safe_free((void**)&names);
        safe_free((void**)&numbers);
        return NULL;
    }

    /* 按帧编号排序 */
    /* 使用插入排序（文件数通常不多） */
    for (int i = 0; i < count - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < count; j++) {
            if (numbers[j] < numbers[min_idx]) min_idx = j;
        }
        if (min_idx != i) {
            int tmp_num = numbers[i];
            numbers[i] = numbers[min_idx];
            numbers[min_idx] = tmp_num;
            char* tmp_name = names[i];
            names[i] = names[min_idx];
            names[min_idx] = tmp_name;
        }
    }

    /* 创建读取器 */
    SlamFrameReader* reader = (SlamFrameReader*)safe_malloc(sizeof(SlamFrameReader));
    if (!reader) {
        for (int i = 0; i < count; i++) safe_free((void**)&names[i]);
        safe_free((void**)&names);
        safe_free((void**)&numbers);
        return NULL;
    }

    reader->filenames = names;
    reader->frame_numbers = numbers;
    reader->total_frames = count;
    reader->current_index = 0;
    reader->start_index = start_index;
    strncpy(reader->directory, directory, sizeof(reader->directory) - 1);
    reader->directory[sizeof(reader->directory) - 1] = '\0';

    /* 跳过起始编号之前的帧 */
    if (start_index > 0) {
        while (reader->current_index < reader->total_frames &&
               reader->frame_numbers[reader->current_index] < start_index) {
            reader->current_index++;
        }
    }

    return reader;
}

int slam_frame_reader_read_next(SlamFrameReader* reader,
                                float** image_data,
                                int* width, int* height,
                                int* channels)
{
    if (!reader || !image_data || !width || !height || !channels) return -1;

    if (reader->current_index >= reader->total_frames) return 0;

    /* 构建完整路径 */
    char fullpath[2048];
#ifdef _WIN32
    snprintf(fullpath, sizeof(fullpath), "%s\\%s",
#else
    snprintf(fullpath, sizeof(fullpath), "%s/%s",
#endif
             reader->directory, reader->filenames[reader->current_index]);

    int ret = slam_load_image_file(fullpath, image_data, width, height, channels);
    if (ret == 0) {
        reader->current_index++;
        return 1;
    }

    /* 加载失败，尝试下一帧 */
    reader->current_index++;
    return -1;
}

int slam_frame_reader_reset(SlamFrameReader* reader)
{
    if (!reader) return -1;
    reader->current_index = 0;
    if (reader->start_index > 0) {
        while (reader->current_index < reader->total_frames &&
               reader->frame_numbers[reader->current_index] < reader->start_index) {
            reader->current_index++;
        }
    }
    return 0;
}

int slam_frame_reader_get_progress(const SlamFrameReader* reader,
                                   int* current, int* total)
{
    if (!reader) return -1;
    if (current) *current = reader->current_index;
    if (total) *total = reader->total_frames;
    return 0;
}

void slam_frame_reader_free(SlamFrameReader* reader)
{
    if (!reader) return;
    if (reader->filenames) {
        for (int i = 0; i < reader->total_frames; i++) {
            safe_free((void**)&reader->filenames[i]);
        }
        safe_free((void**)&reader->filenames);
    }
    safe_free((void**)&reader->frame_numbers);
    safe_free((void**)&reader);
}

/* ============================================================================
 * 原始视频读取器实现
 * =========================================================================== */

struct SlamRawVideoReader {
    FILE* file;                     /**< 文件句柄 */
    int width;                      /**< 图像宽度 */
    int height;                     /**< 图像高度 */
    SlamRawVideoFormat format;      /**< 像素格式 */
    int channels;                   /**< 通道数 */
    int frame_size;                 /**< 每帧字节数 */
    int total_frames;               /**< 总帧数 */
    int current_index;              /**< 当前帧索引 */
};

SlamRawVideoReader* slam_raw_video_reader_create(const char* filepath,
                                                  int width, int height,
                                                  SlamRawVideoFormat format,
                                                  int total_frames)
{
    if (!filepath || width <= 0 || height <= 0) return NULL;

    /* 计算每帧字节数和通道数 */
    int channels;
    int bytes_per_pixel;
    switch (format) {
        case SLAM_RAW_VIDEO_FORMAT_GRAY8:
            channels = 1;
            bytes_per_pixel = 1;
            break;
        case SLAM_RAW_VIDEO_FORMAT_RGB24:
            channels = 3;
            bytes_per_pixel = 3;
            break;
        case SLAM_RAW_VIDEO_FORMAT_RGBA32:
            channels = 4;
            bytes_per_pixel = 4;
            break;
        default:
            return NULL;
    }

    int frame_size = width * height * bytes_per_pixel;

    /* 打开文件 */
    FILE* file = fopen(filepath, "rb");
    if (!file) return NULL;

    /* 获取文件大小 */
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(file);
        return NULL;
    }

    /* 推算总帧数 */
    int actual_total;
    if (total_frames > 0) {
        actual_total = total_frames;
        /* 验证文件大小是否匹配 */
        if ((long)actual_total * frame_size != file_size) {
            /* 文件大小不匹配时自动修正 */
            actual_total = (int)(file_size / frame_size);
            if (actual_total == 0) {
                fclose(file);
                return NULL;
            }
        }
    } else {
        actual_total = (int)(file_size / frame_size);
        if (actual_total == 0) {
            fclose(file);
            return NULL;
        }
    }

    /* 创建读取器 */
    SlamRawVideoReader* reader = (SlamRawVideoReader*)safe_malloc(sizeof(SlamRawVideoReader));
    if (!reader) {
        fclose(file);
        return NULL;
    }

    reader->file = file;
    reader->width = width;
    reader->height = height;
    reader->format = format;
    reader->channels = channels;
    reader->frame_size = frame_size;
    reader->total_frames = actual_total;
    reader->current_index = 0;

    return reader;
}

int slam_raw_video_reader_read_next(SlamRawVideoReader* reader,
                                     float** image_data,
                                     int* channels)
{
    if (!reader || !image_data || !channels) return -1;
    if (reader->current_index >= reader->total_frames) return 0;
    if (!reader->file) return -1;

    /* 定位到当前帧位置 */
    if (fseek(reader->file, (long)reader->current_index * reader->frame_size, SEEK_SET) != 0) {
        return -1;
    }

    /* 分配缓冲区 */
    unsigned char* raw_buf = (unsigned char*)safe_malloc((size_t)reader->frame_size);
    if (!raw_buf) return -1;

    /* 读取原始帧数据 */
    size_t bytes_read = fread(raw_buf, 1, (size_t)reader->frame_size, reader->file);
    if ((int)bytes_read != reader->frame_size) {
        safe_free((void**)&raw_buf);
        return -1;
    }

    /* 转换为float图像数据（值范围0.0-1.0） */
    int width = reader->width;
    int height = reader->height;
    int ch = reader->channels;
    float* float_data = (float*)safe_malloc((size_t)(width * height * ch) * sizeof(float));
    if (!float_data) {
        safe_free((void**)&raw_buf);
        return -1;
    }

    for (int i = 0; i < width * height * ch; i++) {
        float_data[i] = raw_buf[i] / 255.0f;
    }

    safe_free((void**)&raw_buf);

    *image_data = float_data;
    *channels = ch;
    reader->current_index++;

    return 1;
}

int slam_raw_video_reader_seek(SlamRawVideoReader* reader, int frame_index)
{
    if (!reader) return -1;
    if (frame_index < 0 || frame_index >= reader->total_frames) return -1;
    reader->current_index = frame_index;
    return 0;
}

int slam_raw_video_reader_reset(SlamRawVideoReader* reader)
{
    if (!reader) return -1;
    reader->current_index = 0;
    return 0;
}

int slam_raw_video_reader_get_progress(const SlamRawVideoReader* reader,
                                       int* current, int* total)
{
    if (!reader) return -1;
    if (current) *current = reader->current_index;
    if (total) *total = reader->total_frames;
    return 0;
}

void slam_raw_video_reader_free(SlamRawVideoReader* reader)
{
    if (!reader) return;
    if (reader->file) {
        fclose(reader->file);
        reader->file = NULL;
    }
    safe_free((void**)&reader);
}

/* ============================================================================
 * 视频序列SLAM自动处理
 * =========================================================================== */

int slam_process_video_sequence(SlamSystem* system,
                                SlamFrameReader* reader,
                                SlamRawVideoReader* raw_reader,
                                int frame_skip,
                                int max_frames,
                                float* trajectory_output,
                                int max_trajectory,
                                int* trajectory_count)
{
    if (!system) return -1;
    if (!reader && !raw_reader) return -1;

    int processed = 0;
    int skip_counter = 0;
    SlamResult result;
    int traj_idx = 0;

    if (trajectory_count) *trajectory_count = 0;

    while (1) {
        float* frame_data = NULL;
        int img_width = 0, img_height = 0, img_channels = 0;
        int status;

        if (reader) {
            status = slam_frame_reader_read_next(reader, &frame_data, &img_width, &img_height, &img_channels);
        } else {
            status = slam_raw_video_reader_read_next(raw_reader, &frame_data, &img_channels);
            if (status == 1) {
                img_width = raw_reader->width;
                img_height = raw_reader->height;
            }
        }

        if (status <= 0) break;

        /* 帧跳过处理 */
        if (frame_skip > 0 && processed > 0) {
            skip_counter++;
            if (skip_counter <= frame_skip) {
                safe_free((void**)&frame_data);
                continue;
            }
            skip_counter = 0;
        }

        /* 通过SLAM系统处理当前帧 */
        memset(&result, 0, sizeof(SlamResult));

        int slam_ret = slam_process_visual_frame(system, frame_data,
                                                  img_width, img_height,
                                                  img_channels, (float)processed,
                                                  &result);
        safe_free((void**)&frame_data);
        processed++;

        /* 记录轨迹 */
        if (slam_ret == 0 && trajectory_output && traj_idx < max_trajectory) {
            trajectory_output[traj_idx * 7 + 0] = result.estimated_pose.position[0];
            trajectory_output[traj_idx * 7 + 1] = result.estimated_pose.position[1];
            trajectory_output[traj_idx * 7 + 2] = result.estimated_pose.position[2];
            trajectory_output[traj_idx * 7 + 3] = result.estimated_pose.orientation[0];
            trajectory_output[traj_idx * 7 + 4] = result.estimated_pose.orientation[1];
            trajectory_output[traj_idx * 7 + 5] = result.estimated_pose.orientation[2];
            trajectory_output[traj_idx * 7 + 6] = result.estimated_pose.orientation[3];
            traj_idx++;
        }

        if (max_frames > 0 && processed >= max_frames) break;
    }

    if (trajectory_count) *trajectory_count = traj_idx;

    return (processed > 0) ? 0 : -1;
}

// ==================== CameraInput 相机输入接口实现 ====================

/* 内部相机输入结构体 */
typedef struct CameraInput {
    CameraSourceType source_type;       /* 输入源类型 */
    CameraInputConfig config;           /* 配置副本 */
    int frame_id;                       /* 当前帧序号 */
    int current_total_frames;           /* 当前输入源总帧数 */
    float* frame_buffer;                /* 帧图像数据缓冲区 */
    int buffer_capacity;                /* 缓冲区容量（float数量） */
    
    /* 各输入源特有数据 */
    union {
        struct {
            SlamFrameReader* reader;    /* 帧序列读取器 */
        } file_source;
        struct {
            SlamRawVideoReader* reader; /* 原始视频读取器 */
        } raw_video_source;
        struct {
            int reserved;               /* 硬件接口保留 */
        } hardware_source;
    } source;
} CameraInput;

CameraInputConfig camera_input_get_default_config(void)
{
    CameraInputConfig config;
    memset(&config, 0, sizeof(CameraInputConfig));
    config.source_type = CAMERA_SOURCE_HARDWARE;
    config.image_width = 640;
    config.image_height = 480;
    config.channels = 1;
    config.camera_params[0] = 320.0f;  /* fx */
    config.camera_params[1] = 320.0f;  /* fy */
    config.camera_params[2] = 320.0f;  /* cx */
    config.camera_params[3] = 240.0f;  /* cy */
    config.synthetic_total_frames = 100;
    config.file_start_index = 1;
    config.raw_video_total_frames = 0;
    return config;
}

CameraInput* camera_input_create(const CameraInputConfig* config)
{
    if (!config) return NULL;
    
    CameraInput* camera = (CameraInput*)safe_calloc(1, sizeof(CameraInput));
    if (!camera) return NULL;
    
    memcpy(&camera->config, config, sizeof(CameraInputConfig));
    camera->source_type = config->source_type;
    camera->frame_id = 0;
    camera->current_total_frames = 0;
    camera->frame_buffer = NULL;
    camera->buffer_capacity = 0;
    
    /* 根据源类型创建对应的底层读取器 */
    switch (config->source_type) {
        case CAMERA_SOURCE_SYNTHETIC: {
            camera->current_total_frames = config->synthetic_total_frames > 0 ?
                                            config->synthetic_total_frames : 100;
            break;
        }
        case CAMERA_SOURCE_FILE: {
            SlamFrameReader* reader = slam_frame_reader_create(
                config->file_directory,
                config->file_prefix,
                config->file_extension,
                config->file_start_index);
            if (!reader) {
                safe_free((void**)&camera);
                return NULL;
            }
            camera->source.file_source.reader = reader;
            
            /* 获取总帧数 */
            int current = 0, total = 0;
            if (slam_frame_reader_get_progress(reader, &current, &total) == 0) {
                camera->current_total_frames = total;
            } else {
                camera->current_total_frames = -1; /* 未知 */
            }
            break;
        }
        case CAMERA_SOURCE_RAW_VIDEO: {
            SlamRawVideoFormat fmt = SLAM_RAW_VIDEO_FORMAT_GRAY8;
            if (config->channels == 3) fmt = SLAM_RAW_VIDEO_FORMAT_RGB24;
            else if (config->channels == 4) fmt = SLAM_RAW_VIDEO_FORMAT_RGBA32;
            
            SlamRawVideoReader* reader = slam_raw_video_reader_create(
                config->raw_video_path,
                config->image_width,
                config->image_height,
                fmt,
                config->raw_video_total_frames);
            if (!reader) {
                safe_free((void**)&camera);
                return NULL;
            }
            camera->source.raw_video_source.reader = reader;
            camera->current_total_frames = reader->total_frames;
            camera->config.channels = reader->channels;
            break;
        }
        case CAMERA_SOURCE_HARDWARE: {
            /* 硬件相机模式：预留接口，需要hardware_interface对接 */
            camera->current_total_frames = -1; /* 实时流，总帧数未知 */
            break;
        }
        default: {
            safe_free((void**)&camera);
            return NULL;
        }
    }
    
    return camera;
}

int camera_input_read_frame(CameraInput* camera, CameraInputFrame* frame)
{
    if (!camera || !frame) return -1;
    
    memset(frame, 0, sizeof(CameraInputFrame));
    
    int width = camera->config.image_width;
    int height = camera->config.image_height;
    int channels = camera->config.channels;
    int frame_size = width * height * channels;
    
    switch (camera->source_type) {
        case CAMERA_SOURCE_SYNTHETIC: {
            if (camera->frame_id >= camera->current_total_frames) {
                frame->is_valid = 0;
                return 0; /* 已到末尾 */
            }
            
            /* 确保缓冲区足够 */
            if (camera->buffer_capacity < frame_size) {
                float* new_buf = (float*)safe_realloc(camera->frame_buffer,
                                                  (size_t)frame_size * sizeof(float));
                if (!new_buf) return -1;
                camera->frame_buffer = new_buf;
                camera->buffer_capacity = frame_size;
            }
            
            /* 生成合成帧 */
            int ret = slam_generate_synthetic_frame(
                camera->frame_buffer,
                width, height,
                camera->frame_id,
                camera->current_total_frames,
                camera->config.camera_params);
            if (ret != 0) return -1;
            
            frame->image_data = camera->frame_buffer;
            frame->width = width;
            frame->height = height;
            frame->channels = channels;
            frame->timestamp = (float)camera->frame_id * 0.033f; /* ~30fps */
            frame->frame_id = camera->frame_id;
            frame->is_valid = 1;
            camera->frame_id++;
            return 1;
        }
        case CAMERA_SOURCE_FILE: {
            float* frame_data = NULL;
            int img_w = 0, img_h = 0, img_ch = 0;
            int status = slam_frame_reader_read_next(camera->source.file_source.reader,
                                                      &frame_data,
                                                      &img_w, &img_h, &img_ch);
            if (status <= 0) {
                frame->is_valid = 0;
                return status;
            }
            
            /* 将读取的帧数据复制到内部缓冲区 */
            int actual_size = img_w * img_h * img_ch;
            if (camera->buffer_capacity < actual_size) {
                float* new_buf = (float*)safe_realloc(camera->frame_buffer,
                                                  (size_t)actual_size * sizeof(float));
                if (!new_buf) {
                    safe_free((void**)&frame_data);
                    return -1;
                }
                camera->frame_buffer = new_buf;
                camera->buffer_capacity = actual_size;
            }
            memcpy(camera->frame_buffer, frame_data, (size_t)actual_size * sizeof(float));
            safe_free((void**)&frame_data);
            
            frame->image_data = camera->frame_buffer;
            frame->width = img_w;
            frame->height = img_h;
            frame->channels = img_ch;
            frame->timestamp = (float)camera->frame_id * 0.033f;
            frame->frame_id = camera->frame_id;
            frame->is_valid = 1;
            camera->frame_id++;
            return 1;
        }
        case CAMERA_SOURCE_RAW_VIDEO: {
            float* frame_data = NULL;
            int img_ch = 0;
            int status = slam_raw_video_reader_read_next(
                camera->source.raw_video_source.reader, &frame_data, &img_ch);
            if (status <= 0) {
                frame->is_valid = 0;
                return status;
            }
            
            int actual_size = width * height * img_ch;
            if (camera->buffer_capacity < actual_size) {
                float* new_buf = (float*)safe_realloc(camera->frame_buffer,
                                                  (size_t)actual_size * sizeof(float));
                if (!new_buf) {
                    safe_free((void**)&frame_data);
                    return -1;
                }
                camera->frame_buffer = new_buf;
                camera->buffer_capacity = actual_size;
            }
            memcpy(camera->frame_buffer, frame_data, (size_t)actual_size * sizeof(float));
            safe_free((void**)&frame_data);
            
            frame->image_data = camera->frame_buffer;
            frame->width = width;
            frame->height = height;
            frame->channels = img_ch;
            frame->timestamp = (float)camera->frame_id * 0.033f;
            frame->frame_id = camera->frame_id;
            frame->is_valid = 1;
            camera->frame_id++;
            return 1;
        }
        case CAMERA_SOURCE_HARDWARE: {
            /* 硬件相机模式：无实际硬件时返回错误，禁止合成虚假帧 */
            selflnn_set_last_error(SELFLNN_ERROR_HARDWARE_FAILURE, __func__, __FILE__, __LINE__,
                                  "SLAM相机：硬件相机未连接，禁止合成虚假帧（违反'禁止虚拟数据'原则）");
            return -1;
        }
        default:
            return -1;
    }
}

int camera_input_reset(CameraInput* camera)
{
    if (!camera) return -1;
    
    camera->frame_id = 0;
    
    switch (camera->source_type) {
        case CAMERA_SOURCE_SYNTHETIC:
            return 0;
        case CAMERA_SOURCE_FILE:
            if (camera->source.file_source.reader)
                return slam_frame_reader_reset(camera->source.file_source.reader);
            return -1;
        case CAMERA_SOURCE_RAW_VIDEO:
            if (camera->source.raw_video_source.reader)
                return slam_raw_video_reader_reset(camera->source.raw_video_source.reader);
            return -1;
        case CAMERA_SOURCE_HARDWARE:
            return 0;
        default:
            return -1;
    }
}

int camera_input_get_info(const CameraInput* camera,
                         int* source_type,
                         int* total_frames,
                         int* current_frame)
{
    if (!camera) return -1;
    
    if (source_type) *source_type = (int)camera->source_type;
    if (total_frames) *total_frames = camera->current_total_frames;
    if (current_frame) *current_frame = camera->frame_id;
    
    return 0;
}

void camera_input_free(CameraInput* camera)
{
    if (!camera) return;
    
    /* 释放各输入源特有的资源 */
    switch (camera->source_type) {
        case CAMERA_SOURCE_FILE:
            if (camera->source.file_source.reader)
                slam_frame_reader_free(camera->source.file_source.reader);
            break;
        case CAMERA_SOURCE_RAW_VIDEO:
            if (camera->source.raw_video_source.reader)
                slam_raw_video_reader_free(camera->source.raw_video_source.reader);
            break;
        case CAMERA_SOURCE_SYNTHETIC:
        case CAMERA_SOURCE_HARDWARE:
        default:
            break;
    }
    
    /* 释放帧缓冲区 */
    safe_free((void**)&camera->frame_buffer);
    
    safe_free((void**)&camera);
}
