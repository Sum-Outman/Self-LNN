/**
 * @file slam.h
 * @brief 同步定位与建图(SLAM)系统接口
 * 
 * SLAM系统实现完整的同步定位与建图功能，支持视觉SLAM和激光SLAM。
 * 包括视觉里程计、后端优化、地图构建、闭环检测等核心组件。
 * 基于现有的图优化、点云处理、深度估计基础设施构建。
 */

#ifndef SELFLNN_SLAM_H
#define SELFLNN_SLAM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief SLAM系统类型枚举
 */
typedef enum {
    SLAM_TYPE_VISUAL = 0,            /**< 视觉SLAM（单目/双目/RGB-D） */
    SLAM_TYPE_LIDAR = 1,             /**< 激光SLAM（2D/3D激光雷达） */
    SLAM_TYPE_VISUAL_LIDAR = 2,      /**< 视觉-激光融合SLAM */
    SLAM_TYPE_INERTIAL_VISUAL = 3,   /**< 惯性-视觉融合SLAM（VIO） */
} SlamType;

/**
 * @brief SLAM传感器类型枚举
 */
typedef enum {
    SLAM_SENSOR_MONOCULAR = 0,       /**< 单目相机 */
    SLAM_SENSOR_STEREO = 1,          /**< 双目相机 */
    SLAM_SENSOR_RGBD = 2,            /**< RGB-D相机（深度相机） */
    SLAM_SENSOR_LIDAR_2D = 3,        /**< 2D激光雷达 */
    SLAM_SENSOR_LIDAR_3D = 4,        /**< 3D激光雷达（点云） */
    SLAM_SENSOR_IMU = 5,             /**< 惯性测量单元 */
    SLAM_SENSOR_GPS = 6,             /**< GPS定位 */
} SlamSensorType;

/**
 * @brief 视觉里程计方法枚举
 */
typedef enum {
    VO_METHOD_FEATURE_BASED = 0,     /**< 特征点法（ORB、SIFT等） */
    VO_METHOD_DIRECT = 1,            /**< 直接法（光度误差） */
    VO_METHOD_SEMIDIRECT = 2,        /**< 半直接法（SVO） */
    VO_METHOD_HYBRID = 3,            /**< 混合方法 */
} VoMethod;

/**
 * @brief 闭环检测方法枚举
 */
typedef enum {
    LOOP_CLOSURE_METHOD_DISTANCE = 0,    /**< 基于位姿距离的候选选择 */
    LOOP_CLOSURE_METHOD_BOW = 1,         /**< 基于视觉词袋的候选选择 */
    LOOP_CLOSURE_METHOD_HYBRID = 2,      /**< 混合方法（词袋+距离） */
} LoopClosureMethod;

/**
 * @brief 视觉单词结构体（词汇表叶子节点）
 */
typedef struct {
    float descriptor[128];           /**< 单词描述子（聚类中心）128维SIFT */
    int descriptor_length;           /**< 描述子实际长度 */
    float weight;                    /**< TF-IDF权重 */
    int image_count;                 /**< 包含该单词的图像数量 */
} VisualWord;

/**
 * @brief 视觉词汇表配置结构体
 */
typedef struct {
    int vocabulary_size;             /**< 词汇表大小（单词数量） */
    int vocabulary_depth;            /**< 词汇树深度（层次聚类层数） */
    int branching_factor;            /**< 分支因子（每层聚类数） */
    int max_train_frames;            /**< 最大训练帧数 */
    int descriptor_length;           /**< 描述子长度（默认32） */
    int enable_tf_idf;               /**< 是否启用TF-IDF加权 */
    int enable_incremental_update;   /**< 是否启用增量更新 */
} VisualVocabularyConfig;

/**
 * @brief 闭环候选评分结构体
 */
typedef struct {
    int candidate_frame_id;          /**< 候选关键帧ID */
    float bow_score;                 /**< 词袋相似度评分 */
    float geometric_score;           /**< 几何验证评分 */
    float temporal_score;            /**< 时间一致性评分 */
    float total_score;               /**< 综合评分 */
    int num_matches;                 /**< 特征匹配数量 */
    int num_inliers;                 /**< 几何内点数量 */
    float inlier_ratio;              /**< 内点比例 */
    float relative_distance;         /**< 相对位姿距离 */
    int temporal_consistent;         /**< 是否通过时间一致性验证 */
    int verification_passed;         /**< 是否通过全部验证 */
} LoopClosureCandidate;

/**
 * @brief 共视图边结构体
 */
typedef struct {
    int from_frame_id;               /**< 源关键帧ID */
    int to_frame_id;                 /**< 目标关键帧ID */
    int weight;                      /**< 共视权重（共享地标点数） */
    float relative_pose[7];          /**< 相对位姿 (tx,ty,tz,qw,qx,qy,qz) */
} CovisibilityEdge;

/**
 * @brief 闭环检测配置结构体（扩展配置）
 */
typedef struct {
    int enable_temporal_consistency; /**< 是否启用时间一致性验证 */
    int temporal_window_size;        /**< 时间一致性窗口大小 */
    int temporal_consistency_threshold; /**< 时间一致性阈值（连续通过帧数） */
    
    int enable_geometric_verification; /**< 是否启用几何验证 */
    float geometric_threshold;       /**< 几何验证阈值（像素） */
    int min_geometric_inliers;       /**< 最小几何内点数 */
    float min_inlier_ratio;          /**< 最小内点比例 */
    
    int enable_map_fusion;           /**< 是否启用地图融合 */
    int enable_drift_propagation;    /**< 是否启用漂移传播 */
    
    float max_loop_distance;         /**< 最大闭环距离（米） */
    int min_feature_matches;         /**< 最小特征匹配数 */
    int min_keyframes_for_detection; /**< 开始检测的最小关键帧数 */
    int min_frame_gap;               /**< 检测时间窗口间隔 */
    
    int max_candidates_per_detection; /**< 每次检测最大候选数量 */
    float candidate_search_radius;   /**< 候选搜索半径（米） */
    float bow_score_threshold;       /**< 词袋评分阈值 */
} LoopClosureConfig;

/**
 * @brief 后端优化方法枚举
 */
typedef enum {
    BACKEND_OPTIMIZATION_G2O = 0,    /**< 图优化（g2o风格） */
    BACKEND_OPTIMIZATION_GTSAM = 1,  /**< 因子图优化（GTSAM风格） */
    BACKEND_OPTIMIZATION_CERES = 2,  /**< 非线性优化（Ceres风格） */
} BackendOptimizationMethod;

/**
 * @brief 地图表示方法枚举
 */
typedef enum {
    MAP_REPRESENTATION_FEATURE_MAP = 0,  /**< 特征点地图 */
    MAP_REPRESENTATION_POINT_CLOUD = 1,  /**< 点云地图 */
    MAP_REPRESENTATION_OCCUPANCY_GRID = 2, /**< 占用栅格地图（2D） */
    MAP_REPRESENTATION_OCTOMAP = 3,      /**< 八叉树地图（3D） */
    MAP_REPRESENTATION_SEMANTIC = 4,     /**< 语义地图 */
} MapRepresentation;

/**
 * @brief 位姿结构体（6自由度：位置 + 旋转四元数）
 */
typedef struct {
    float position[3];               /**< 位置向量 (x, y, z) */
    float orientation[4];            /**< 旋转四元数 (w, x, y, z) */
    float timestamp;                 /**< 时间戳（秒） */
    int frame_id;                    /**< 帧ID */
} SlamPose;

/**
 * @brief 关键帧结构体
 */
typedef struct {
    int id;                          /**< 关键帧ID */
    SlamPose pose;                   /**< 关键帧位姿 */
    float* image_data;               /**< 图像数据（可选） */
    float* depth_data;               /**< 深度数据（可选） */
    float* point_cloud;              /**< 点云数据（可选） */
    int image_width;                 /**< 图像宽度 */
    int image_height;                /**< 图像高度 */
    size_t point_count;              /**< 点云点数 */
    float* descriptors;              /**< 特征描述子 */
    int num_features;                /**< 特征点数 */
    int* keypoints_x;                /**< 特征点X坐标 */
    int* keypoints_y;                /**< 特征点Y坐标 */
    float* keypoints_3d;             /**< 3D特征点坐标（每点3个浮点数） */
    int* landmark_ids;               /**< 关联的地标点ID数组 */
    int num_landmarks;               /**< 关联的地标点数量 */
} KeyFrame;

/**
 * @brief 地标点结构体
 */
typedef struct {
    int id;                          /**< 地标点ID */
    int is_valid;                    /**< 是否有效（用于前端过滤） */
    float position[3];               /**< 3D位置 (x, y, z) */
    float* descriptor;               /**< 描述子向量（堆分配） */
    int descriptor_length;           /**< 描述子长度 */
    int observed_count;              /**< 被观测次数（实际计数） */
    int num_observations;            /**< 被观测次数（等效于observed_count） */
    int* observing_frames;           /**< 观测到此地标的关键帧ID数组 */
    int* feature_indices;            /**< 每个观测对应的特征点在帧中的索引 */
    float* observations;             /**< 观测到的2D坐标数组（每观测2个浮点数） */
} Landmark;

/**
 * @brief SLAM系统配置结构体
 */
typedef struct {
    SlamType type;                   /**< SLAM系统类型 */
    SlamSensorType sensor_type;      /**< 传感器类型 */
    int image_width;                 /**< 图像宽度（像素） */
    int image_height;                /**< 图像高度（像素） */
    float camera_params[4];          /**< 相机内参：fx, fy, cx, cy */
    VoMethod vo_method;              /**< 视觉里程计方法 */
    BackendOptimizationMethod backend_method; /**< 后端优化方法 */
    MapRepresentation map_representation; /**< 地图表示方法 */
    
    int enable_loop_closure;         /**< 是否启用闭环检测 */
    int enable_online_mapping;       /**< 是否启用在线建图 */
    int enable_localization_only;    /**< 仅定位模式（使用已有地图） */
    int enable_visualization;        /**< 是否启用可视化 */
    
    int max_keyframes;               /**< 最大关键帧数量 */
    int min_keyframe_interval;       /**< 关键帧最小间隔（帧数） */
    float min_keyframe_distance;     /**< 关键帧最小距离（米） */
    float min_keyframe_rotation;     /**< 关键帧最小旋转（弧度） */
    
    int feature_extractor_type;      /**< 特征提取器类型：0=ORB，1=SIFT，2=SURF */
    int num_features_per_frame;      /**< 每帧提取的特征点数量 */
    float feature_scale_factor;      /**< 特征金字塔尺度因子 */
    int feature_pyramid_levels;      /**< 特征金字塔层数 */
    
    float loop_closure_threshold;    /**< 闭环检测阈值 */
    int loop_closure_min_inliers;    /**< 闭环最小内点数 */
    float loop_closure_distance_threshold; /**< 闭环距离阈值（米） */
    
    int use_gpu;                     /**< 是否使用GPU加速 */
    int thread_count;                /**< 并行线程数量 */
    
    int use_depth;                   /**< 是否使用深度信息 */
    float voxel_size;                /**< 体素大小（点云下采样） */
    
    float mapping_resolution;        /**< 地图分辨率（米） */
    float occupancy_threshold;       /**< 占用概率阈值 */
    
    float imu_noise_gyro;            /**< IMU陀螺仪噪声 */
    float imu_noise_acc;             /**< IMU加速度计噪声 */
    float imu_bias_noise;            /**< IMU偏置噪声 */
} SlamConfig;

/**
 * @brief SLAM系统状态结构体
 */
typedef struct {
    SlamPose current_pose;           /**< 当前位姿 */
    int tracking_state;              /**< 跟踪状态：0=丢失，1=弱，2=良好 */
    int mapping_state;               /**< 建图状态：0=未建图，1=正在建图，2=已建图 */
    int loop_closure_state;          /**< 闭环状态：0=未检测，1=检测中，2=已检测 */
    
    int total_frames;                /**< 总处理帧数 */
    int total_keyframes;             /**< 总关键帧数 */
    int total_landmarks;             /**< 总地标点数 */
    
    float tracking_time_ms;          /**< 跟踪时间（毫秒） */
    float mapping_time_ms;           /**< 建图时间（毫秒） */
    float optimization_time_ms;      /**< 优化时间（毫秒） */
    
    float map_coverage_area;         /**< 地图覆盖面积（平方米） */
    float map_accuracy;              /**< 地图精度（米） */
    float localization_accuracy;     /**< 定位精度（米） */
    float reprojection_error;        /**< 平均重投影误差（像素） */
    int loop_closures_detected;      /**< 检测到的闭环数量 */
} SlamState;

/**
 * @brief SLAM处理结果结构体
 */
typedef struct {
    SlamPose estimated_pose;         /**< 估计位姿 */
    int tracking_quality;            /**< 跟踪质量：0-100 */
    int new_keyframe_created;        /**< 是否创建了新关键帧 */
    int loop_closure_detected;       /**< 是否检测到闭环 */
    int landmarks_updated;           /**< 是否更新了地标点 */
    
    float* local_map_points;         /**< 局部地图点云 */
    size_t local_map_point_count;    /**< 局部地图点数 */
    
    float* trajectory;               /**< 轨迹历史（每帧位姿） */
    int trajectory_length;           /**< 轨迹长度 */
    
    float processing_time_ms;        /**< 处理时间（毫秒） */
    float timestamp;                 /**< 时间戳 */
} SlamResult;

/**
 * @brief SLAM系统句柄
 */
typedef struct SlamSystem SlamSystem;

/**
 * @brief 获取默认的SLAM配置
 * 
 * @return SlamConfig 默认配置
 */
SlamConfig slam_get_default_config(void);

/**
 * @brief 创建SLAM系统
 * 
 * @param config SLAM配置参数
 * @return SlamSystem* SLAM系统句柄，失败返回NULL
 */
SlamSystem* slam_system_create(const SlamConfig* config);

/**
 * @brief 释放SLAM系统
 * 
 * @param system SLAM系统句柄
 */
void slam_system_free(SlamSystem* system);

/**
 * @brief 处理新的传感器数据帧
 * 
 * @param system SLAM系统句柄
 * @param sensor_data 传感器数据（图像、点云等）
 * @param sensor_type 传感器类型
 * @param width 数据宽度（图像宽度或点云点数）
 * @param height 数据高度（图像高度，点云为1）
 * @param channels 通道数（图像通道数，点云为3或6）
 * @param timestamp 时间戳（秒）
 * @param result 处理结果输出
 * @return int 成功返回0，失败返回-1
 */
int slam_process_frame(SlamSystem* system,
                      const float* sensor_data,
                      SlamSensorType sensor_type,
                      int width, int height, int channels,
                      float timestamp,
                      SlamResult* result);

/**
 * @brief 处理视觉帧（相机图像）
 * 
 * @param system SLAM系统句柄
 * @param image_data 图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数（1=灰度，3=RGB）
 * @param timestamp 时间戳
 * @param result 处理结果输出
 * @return int 成功返回0，失败返回-1
 */
int slam_process_visual_frame(SlamSystem* system,
                             const float* image_data,
                             int width, int height, int channels,
                             float timestamp,
                             SlamResult* result);

/**
 * @brief 处理点云帧（激光雷达）
 * 
 * @param system SLAM系统句柄
 * @param point_cloud 点云数据（每点3或6个浮点数：XYZ或XYZ+RGB）
 * @param point_count 点数
 * @param timestamp 时间戳
 * @param result 处理结果输出
 * @return int 成功返回0，失败返回-1
 */
int slam_process_point_cloud(SlamSystem* system,
                            const float* point_cloud,
                            size_t point_count,
                            float timestamp,
                            SlamResult* result);

/**
 * @brief 处理IMU数据
 * 
 * @param system SLAM系统句柄
 * @param gyro 陀螺仪数据（3轴，弧度/秒）
 * @param acc 加速度计数据（3轴，米/秒²）
 * @param timestamp 时间戳
 * @return int 成功返回0，失败返回-1
 */
int slam_process_imu_data(SlamSystem* system,
                         const float gyro[3],
                         const float acc[3],
                         float timestamp);

/**
 * @brief 重置SLAM系统
 * 
 * @param system SLAM系统句柄
 * @param reset_map 是否重置地图
 */
void slam_system_reset(SlamSystem* system, int reset_map);

/**
 * @brief 获取当前SLAM状态
 * 
 * @param system SLAM系统句柄
 * @param state 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int slam_get_state(const SlamSystem* system, SlamState* state);

/**
 * @brief 获取SLAM估计轨迹
 * 
 * @param system SLAM系统句柄
 * @param poses 轨迹位姿输出缓冲区（为NULL时返回轨迹长度）
 * @param max_count 输出缓冲区最大容量
 * @return int 轨迹中位姿数量
 */
int slam_get_trajectory(const SlamSystem* system, SlamPose* poses, int max_count);

/**
 * @brief 获取SLAM系统配置
 * 
 * @param system SLAM系统句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int slam_get_config(const SlamSystem* system, SlamConfig* config);

/**
 * @brief 更新SLAM系统配置
 * 
 * @param system SLAM系统句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int slam_set_config(SlamSystem* system, const SlamConfig* config);

/**
 * @brief 保存SLAM地图到文件
 * 
 * @param system SLAM系统句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int slam_save_map(const SlamSystem* system, const char* filepath);

/**
 * @brief 从文件加载SLAM地图
 * 
 * @param system SLAM系统句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int slam_load_map(SlamSystem* system, const char* filepath);

/**
 * @brief 保存SLAM轨迹到文件
 * 
 * @param system SLAM系统句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int slam_save_trajectory(const SlamSystem* system, const char* filepath);

/**
 * @brief 获取完整地图点云
 * 
 * @param system SLAM系统句柄
 * @param point_cloud 点云输出缓冲区
 * @param max_points 最大点数
 * @return int 成功返回点数，失败返回-1
 */
int slam_get_map_point_cloud(const SlamSystem* system,
                            float* point_cloud,
                            size_t max_points);

/**
 * @brief 获取所有关键帧
 * 
 * @param system SLAM系统句柄
 * @param keyframes 关键帧输出缓冲区
 * @param max_keyframes 最大关键帧数
 * @return int 成功返回关键帧数，失败返回-1
 */
int slam_get_keyframes(const SlamSystem* system,
                      KeyFrame* keyframes,
                      int max_keyframes);

/**
 * @brief 获取所有地标点
 * 
 * @param system SLAM系统句柄
 * @param landmarks 地标点输出缓冲区
 * @param max_landmarks 最大地标点数
 * @return int 成功返回地标点数，失败返回-1
 */
int slam_get_landmarks(const SlamSystem* system,
                      Landmark* landmarks,
                      int max_landmarks);

/**
 * @brief 执行全局优化（捆集调整）
 * 
 * @param system SLAM系统句柄
 * @param iterations 优化迭代次数
 * @return int 成功返回0，失败返回-1
 */
int slam_perform_global_optimization(SlamSystem* system, int iterations);

/**
 * @brief 执行局部优化
 * 
 * @param system SLAM系统句柄
 * @param window_size 优化窗口大小（关键帧数）
 * @param iterations 优化迭代次数
 * @return int 成功返回0，失败返回-1
 */
int slam_perform_local_optimization(SlamSystem* system, int window_size, int iterations);

/**
 * @brief 手动触发闭环检测
 * 
 * @param system SLAM系统句柄
 * @param candidate_frame_id 候选关键帧ID
 * @return int 成功返回匹配的关键帧ID，失败返回-1
 */
int slam_trigger_loop_closure(SlamSystem* system, int candidate_frame_id);

/**
 * @brief 获取可视化数据
 * 
 * @param system SLAM系统句柄
 * @param visualization_data 可视化数据输出缓冲区
 * @param max_size 最大数据大小（字节）
 * @return int 成功返回数据大小，失败返回-1
 */
int slam_get_visualization_data(const SlamSystem* system,
                               unsigned char* visualization_data,
                               size_t max_size);

/**
 * @brief 设置初始位姿
 * 
 * @param system SLAM系统句柄
 * @param initial_pose 初始位姿
 * @return int 成功返回0，失败返回-1
 */
int slam_set_initial_pose(SlamSystem* system, const SlamPose* initial_pose);

/**
 * @brief 获取当前位姿协方差
 * 
 * @param system SLAM系统句柄
 * @param covariance 协方差矩阵输出（6x6，行主序）
 * @return int 成功返回0，失败返回-1
 */
int slam_get_pose_covariance(const SlamSystem* system, float* covariance);

/**
 * @brief 获取地图质量指标
 * 
 * @param system SLAM系统句柄
 * @param consistency 一致性指标输出
 * @param completeness 完整性指标输出
 * @param accuracy 精度指标输出
 * @return int 成功返回0，失败返回-1
 */
int slam_get_map_quality(const SlamSystem* system,
                        float* consistency,
                        float* completeness,
                        float* accuracy);

/**
 * @brief 设置闭环检测扩展配置
 * 
 * @param system SLAM系统句柄
 * @param config 闭环检测配置
 * @return int 成功返回0，失败返回-1
 */
int slam_set_loop_closure_config(SlamSystem* system, const LoopClosureConfig* config);

/**
 * @brief 获取闭环检测扩展配置
 * 
 * @param system SLAM系统句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int slam_get_loop_closure_config(const SlamSystem* system, LoopClosureConfig* config);

/**
 * @brief 设置视觉词汇表配置
 * 
 * @param system SLAM系统句柄
 * @param config 词汇表配置
 * @return int 成功返回0，失败返回-1
 */
int slam_set_vocabulary_config(SlamSystem* system, const VisualVocabularyConfig* config);

/**
 * @brief 训练/构建视觉词汇表
 * 
 * @param system SLAM系统句柄
 * @return int 成功返回0，失败返回-1
 */
int slam_build_vocabulary(SlamSystem* system);

/**
 * @brief 增量更新视觉词汇表
 * 
 * @param system SLAM系统句柄
 * @return int 成功返回0，失败返回-1
 */
int slam_update_vocabulary(SlamSystem* system);

/**
 * @brief 获取闭环检测的详细候选列表
 * 
 * @param system SLAM系统句柄
 * @param candidates 候选数组输出缓冲区
 * @param max_count 最大候选数量
 * @return int 成功返回候选数量，失败返回-1
 */
int slam_get_loop_closure_candidates(const SlamSystem* system,
                                    LoopClosureCandidate* candidates,
                                    int max_count);

/**
 * @brief 获取闭环检测质量评分
 * 
 * @param system SLAM系统句柄
 * @param overall_score 综合评分输出
 * @param precision 精度评分输出
 * @param recall 召回率评分输出
 * @return int 成功返回0，失败返回-1
 */
int slam_get_loop_closure_quality(const SlamSystem* system,
                                 float* overall_score,
                                 float* precision,
                                 float* recall);

/**
 * @brief 手动执行地图点融合
 * 
 * @param system SLAM系统句柄
 * @return int 成功返回融合的地图点数，失败返回-1
 */
int slam_fuse_map_points(SlamSystem* system);

/**
 * @brief 获取共视图边列表
 * 
 * @param system SLAM系统句柄
 * @param edges 共视图边输出缓冲区
 * @param max_edges 最大边数
 * @return int 成功返回边数，失败返回-1
 */
int slam_get_covisibility_edges(const SlamSystem* system,
                               CovisibilityEdge* edges,
                               int max_edges);

/**
 * @brief 获取视觉词汇表统计信息
 * 
 * @param system SLAM系统句柄
 * @param vocab_size 词汇表大小输出
 * @param total_words 总单词数输出
 * @param avg_weight 平均权重输出
 * @return int 成功返回0，失败返回-1
 */
int slam_get_vocabulary_stats(const SlamSystem* system,
                            int* vocab_size,
                            int* total_words,
                            float* avg_weight);

/**
 * @brief 生成合成SLAM测试帧图像
 * 
 * 生成模拟相机运动的合成图像序列，包含可跟踪的视觉特征（圆形网格）。
 * 图像模拟一个包含3D特征点的场景，相机沿圆形轨迹运动。
 * 用于SLAM系统的测试和验证，无需实际相机硬件。
 * 
 * @param image_data 图像数据输出缓冲区（需预分配 width*height 个float）
 * @param width 图像宽度（像素）
 * @param height 图像高度（像素）
 * @param frame_id 当前帧序号（从0开始）
 * @param total_frames 总帧数
 * @param camera_params 相机内参 [fx, fy, cx, cy]
 * @return int 成功返回0，失败返回-1
 * @deprecated F-048: 此函数已在严格真实数据模式下禁用，保留声明仅用于API兼容性
 * 
 * 注意: 在SELFLNN_STRICT_REAL_DATA模式下此函数不执行任何操作，
 * 始终返回-1错误码。使用真实相机数据代替。
 */
int slam_generate_synthetic_frame(float* image_data,
                                 int width, int height,
                                 int frame_id, int total_frames,
                                 const float* camera_params);

/**

// ==================== 视频/图像文件加载接口 ====================

/**
 * @brief 图像文件类型枚举
 */
typedef enum {
    SLAM_IMAGE_FORMAT_UNKNOWN = 0,
    SLAM_IMAGE_FORMAT_PPM,          /**< 彩色PPM格式 */
    SLAM_IMAGE_FORMAT_PGM,          /**< 灰度PGM格式 */
    SLAM_IMAGE_FORMAT_RAW_FLOAT,    /**< 原始float数组（无头文件，连续存储） */
    SLAM_IMAGE_FORMAT_RAW_UINT8,    /**< 原始uint8数组（无头文件） */
} SlamImageFormat;

/**
 * @brief 从图像文件加载图像数据
 *
 * 支持PPM/PGM/RAW格式的图像文件，不依赖任何第三方库。
 * PPM/PGM格式基于Netpbm标准，可被多数图像软件导出。
 * 调用者负责释放image_data缓冲区。
 *
 * @param filepath 图像文件路径
 * @param image_data 图像数据输出缓冲区指针（需调用safe_free释放）
 * @param width 图像宽度输出
 * @param height 图像高度输出
 * @param channels 图像通道数输出
 * @return int 成功返回0，失败返回-1
 */
int slam_load_image_file(const char* filepath,
                         float** image_data,
                         int* width, int* height,
                         int* channels);

/**
 * @brief 保存图像数据到PPM/PGM文件
 *
 * 将float图像数据数组保存为标准PPM（彩色）或PGM（灰度）格式。
 * 保存的文件可以被多数图像查看器打开。
 *
 * @param filepath 输出文件路径
 * @param image_data 图像数据（float数组，值范围0.0-1.0）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数（1=灰度PGM，3=RGB彩色PPM）
 * @return int 成功返回0，失败返回-1
 */
int slam_save_image_file(const char* filepath,
                         const float* image_data,
                         int width, int height,
                         int channels);

/**
 * @brief 帧读取器句柄（不透明）
 *
 * 用于按顺序读取一系列编号图像文件，模拟视频帧输入。
 * 文件名格式：{prefix}{frame_id:06d}.{ext}
 * 例如：frame_000001.ppm, frame_000002.ppm, ...
 */
typedef struct SlamFrameReader SlamFrameReader;

/**
 * @brief 创建帧序列读取器
 *
 * 从磁盘上顺序编号的图像文件创建帧读取器。
 * 帧读取器会扫描目录，找到匹配{prefix}{数字}.{ext}模式的所有文件，
 * 并按文件名排序后顺序读取。
 *
 * @param directory 图像文件所在目录（UTF-8路径）
 * @param file_prefix 文件名前缀（如"frame_"）
 * @param file_extension 文件扩展名（如"ppm"、"pgm"）
 * @param start_index 起始编号（如1）
 * @return SlamFrameReader* 帧读取器句柄，失败返回NULL
 */
SlamFrameReader* slam_frame_reader_create(const char* directory,
                                          const char* file_prefix,
                                          const char* file_extension,
                                          int start_index);

/**
 * @brief 读取下一帧图像
 *
 * 从帧读取器中读取下一帧图像数据。
 * 每调用一次，内部帧计数器加1。
 *
 * @param reader 帧读取器句柄
 * @param image_data 图像数据输出缓冲区指针（需调用safe_free释放）
 * @param width 图像宽度输出
 * @param height 图像高度输出
 * @param channels 图像通道数输出
 * @return int 成功返回1（还有更多帧），返回0（已到末尾），失败返回-1
 */
int slam_frame_reader_read_next(SlamFrameReader* reader,
                                float** image_data,
                                int* width, int* height,
                                int* channels);

/**
 * @brief 重置帧读取器到起始位置
 *
 * @param reader 帧读取器句柄
 * @return int 成功返回0，失败返回-1
 */
int slam_frame_reader_reset(SlamFrameReader* reader);

/**
 * @brief 获取帧读取器当前进度
 *
 * @param reader 帧读取器句柄
 * @param current 当前帧索引输出
 * @param total 总帧数输出
 * @return int 成功返回0，失败返回-1
 */
int slam_frame_reader_get_progress(const SlamFrameReader* reader,
                                   int* current, int* total);

/**
 * @brief 释放帧读取器
 *
 * @param reader 帧读取器句柄
 */
void slam_frame_reader_free(SlamFrameReader* reader);

/**
 * @brief 原始视频像素格式枚举
 */
typedef enum {
    SLAM_RAW_VIDEO_FORMAT_GRAY8 = 0,    /**< 8位灰度，每帧 width*height 字节 */
    SLAM_RAW_VIDEO_FORMAT_RGB24 = 1,    /**< 24位RGB，每帧 width*height*3 字节 */
    SLAM_RAW_VIDEO_FORMAT_RGBA32 = 2,   /**< 32位RGBA，每帧 width*height*4 字节 */
} SlamRawVideoFormat;

/**
 * @brief 原始视频读取器句柄（不透明）
 *
 * 从单个二进制文件中按顺序读取原始图像帧数据。
 * 文件必须包含连续的帧数据，每帧大小固定：
 *   GRAY8:  width * height
 *   RGB24:  width * height * 3
 *   RGBA32: width * height * 4
 * 文件总大小必须为 帧数 × 每帧字节数。
 */
typedef struct SlamRawVideoReader SlamRawVideoReader;

/**
 * @brief 创建原始视频读取器
 *
 * 从单个二进制视频文件创建帧读取器。
 * 支持GRAY8/RGB24/RGBA32格式的无压缩原始视频数据。
 *
 * @param filepath 视频文件路径（UTF-8路径）
 * @param width 图像宽度（像素）
 * @param height 图像高度（像素）
 * @param format 像素格式（SlamRawVideoFormat）
 * @param total_frames 视频总帧数（0表示自动从文件大小推算）
 * @return SlamRawVideoReader* 视频读取器句柄，失败返回NULL
 */
SlamRawVideoReader* slam_raw_video_reader_create(const char* filepath,
                                                  int width, int height,
                                                  SlamRawVideoFormat format,
                                                  int total_frames);

/**
 * @brief 读取原始视频下一帧
 *
 * @param reader 视频读取器句柄
 * @param image_data 图像数据输出缓冲区指针（float数组，值范围0.0-1.0，需safe_free释放）
 * @param channels 通道数输出
 * @return int 成功返回1（还有更多帧），返回0（已到末尾），失败返回-1
 */
int slam_raw_video_reader_read_next(SlamRawVideoReader* reader,
                                     float** image_data,
                                     int* channels);

/**
 * @brief 跳转到原始视频的指定帧
 *
 * @param reader 视频读取器句柄
 * @param frame_index 目标帧索引（从0开始）
 * @return int 成功返回0，失败返回-1
 */
int slam_raw_video_reader_seek(SlamRawVideoReader* reader, int frame_index);

/**
 * @brief 重置原始视频读取器到起始位置
 *
 * @param reader 视频读取器句柄
 * @return int 成功返回0，失败返回-1
 */
int slam_raw_video_reader_reset(SlamRawVideoReader* reader);

/**
 * @brief 获取原始视频读取器当前进度
 *
 * @param reader 视频读取器句柄
 * @param current 当前帧索引输出
 * @param total 总帧数输出
 * @return int 成功返回0，失败返回-1
 */
int slam_raw_video_reader_get_progress(const SlamRawVideoReader* reader,
                                       int* current, int* total);

/**
 * @brief 释放原始视频读取器
 *
 * @param reader 视频读取器句柄
 */
void slam_raw_video_reader_free(SlamRawVideoReader* reader);

/**
 * @brief 处理完整视频序列的SLAM建图
 *
 * 从帧读取器中按顺序读取所有图像帧，依次通过SLAM流水线处理。
 * 支持帧读取器和原始视频读取器两种输入源。
 * 处理完成后，整个轨迹和地图将保存在SLAM系统内部。
 *
 * @param system SLAM系统句柄
 * @param reader 帧序列读取器句柄（图像帧源）
 * @param raw_reader 原始视频读取器句柄（视频帧源，与reader二选一，可传NULL）
 * @param frame_skip 帧跳过数（0=每帧都处理，1=隔帧处理，以此类推）
 * @param max_frames 最大处理帧数（0=处理所有帧）
 * @param trajectory_output 轨迹输出缓冲区（每帧7个float: tx,ty,tz,qw,qx,qy,qz，需预分配 max_frames*7*sizeof(float)）
 * @param max_trajectory 轨迹缓冲区容量（帧数）
 * @param trajectory_count 实际轨迹长度输出
 * @return int 成功返回0，失败返回-1
 */
int slam_process_video_sequence(SlamSystem* system,
                                SlamFrameReader* reader,
                                SlamRawVideoReader* raw_reader,
                                int frame_skip,
                                int max_frames,
                                float* trajectory_output,
                                int max_trajectory,
                                int* trajectory_count);

// ==================== CfC直接法视觉里程计增强 ====================

/**
 * @brief CfC直接法VO配置结构体
 */
typedef struct {
    int image_pyramid_levels;        /**< 图像金字塔层数（默认3） */
    int max_iterations_per_level;    /**< 每层最大迭代次数（默认50） */
    float min_update_norm;           /**< 最小更新范数，收敛阈值（默认1e-4） */
    float outlier_threshold;         /**< 光度误差离群值阈值（像素值，默认0.2） */
    int patch_half_size;             /**< 图像块半尺寸（默认2，即5x5块） */
    int cfc_hidden_size;             /**< CfC隐藏状态维度（默认64） */
    float cfc_time_constant;         /**< CfC时间常数（默认0.1） */
    float cfc_noise_std;             /**< CfC过程噪声标准差（默认1e-6） */
    int enable_photometric_calibration; /**< 是否启用光度标定（默认0） */
    int enable_motion_blur_compensation; /**< 是否启用运动模糊补偿（默认0） */
    int enable_adaptive_iteration;   /**< 是否启用在每次金字塔级别中自适应迭代次数（默认1） */
    int enable_feature_support;      /**< 是否结合特征点提高鲁棒性（默认0） */
} CfCDirectVOConfig;

/**
 * @brief CfC直接法VO句柄（不透明）
 *
 * 基于CfC液态神经网络的直接法视觉里程计。
 * 使用CfC ODE连续时间动态系统演化光度误差，
 * 实现帧间位姿估计，替代传统直接法的数值优化。
 * 核心思想：光度误差→CfC隐藏状态→姿态增量。
 */
typedef struct CfCDirectVO CfCDirectVO;

/**
 * @brief 获取默认CfC直接法VO配置
 *
 * @return CfCDirectVOConfig 默认配置
 */
#ifdef SELFLNN_SLAM_ADVANCED
CfCDirectVOConfig slam_cfc_direct_vo_get_default_config(void);
#else
static inline CfCDirectVOConfig slam_cfc_direct_vo_get_default_config(void) { CfCDirectVOConfig c; memset(&c, 0, sizeof(c)); return c; }
#endif

/**
 * @brief 创建CfC直接法VO
 *
 * @param config CfC直接法VO配置
 * @return CfCDirectVO* 句柄，失败返回NULL
 */
#ifdef SELFLNN_SLAM_ADVANCED
CfCDirectVO* slam_cfc_direct_vo_create(const CfCDirectVOConfig* config);
#else
static inline CfCDirectVO* slam_cfc_direct_vo_create(const CfCDirectVOConfig* config) { (void)config; return NULL; }
#endif

/**
 * @brief 释放CfC直接法VO
 *
 * @param vo CfC直接法VO句柄
 */
#ifdef SELFLNN_SLAM_ADVANCED
void slam_cfc_direct_vo_free(CfCDirectVO* vo);
#else
static inline void slam_cfc_direct_vo_free(CfCDirectVO* vo) { (void)vo; }
#endif

/**
 * @brief 处理一对连续帧，估计相对位姿
 *
 * 核心算法：
 * 1. 构建参考帧和当前帧的图像金字塔
 * 2. 对每层金字塔，计算光度误差（参考帧像素投影到当前帧的灰度差异）
 * 3. 将光度误差输入CfC ODE细胞，演化隐藏状态
 * 4. CfC隐藏状态解码为6-DOF位姿增量（tx,ty,tz,qx,qy,qz,qw）
 * 5. 使用逆合成(Inverse Compositional)方式更新位姿估计
 * 6. 金字塔细化：从顶层到底层逐层精化
 *
 * @param vo CfC直接法VO句柄
 * @param ref_image 参考帧图像（float数组，值范围0.0-1.0）
 * @param curr_image 当前帧图像（float数组，值范围0.0-1.0）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数（1=灰度，3=RGB自动转灰度）
 * @param camera_params 相机内参 [fx, fy, cx, cy]
 * @param pose_out 输出位姿 [tx,ty,tz,qw,qx,qy,qz]（参考帧到当前帧的变换）
 * @return int 成功返回0，失败返回-1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_cfc_direct_vo_estimate_pose(CfCDirectVO* vo,
                                     const float* ref_image,
                                     const float* curr_image,
                                     int width, int height, int channels,
                                     const float camera_params[4],
                                     float pose_out[7]);
#else
static inline int slam_cfc_direct_vo_estimate_pose(CfCDirectVO* vo,
                                     const float* ref_image,
                                     const float* curr_image,
                                     int width, int height, int channels,
                                     const float camera_params[4],
                                     float pose_out[7]) { (void)vo; (void)ref_image; (void)curr_image; (void)width; (void)height; (void)channels; (void)camera_params; (void)pose_out; return -1; }
#endif

/**
 * @brief 重置CfC直接法VO的内部状态
 *
 * 清除CfC隐藏状态和金字塔缓存。
 *
 * @param vo CfC直接法VO句柄
 */
#ifdef SELFLNN_SLAM_ADVANCED
void slam_cfc_direct_vo_reset(CfCDirectVO* vo);
#else
static inline void slam_cfc_direct_vo_reset(CfCDirectVO* vo) { (void)vo; }
#endif

/**
 * @brief 获取CfC直接法VO的内部状态信息
 *
 * @param vo CfC直接法VO句柄
 * @param iterations_used 输出上次估计使用的迭代次数
 * @param final_error 输出最终光度误差
 * @param cfc_hidden_state 输出CfC隐藏状态（需预分配hidden_size个float）
 * @param hidden_size CfC隐藏状态缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_cfc_direct_vo_get_state(const CfCDirectVO* vo,
                                 int* iterations_used,
                                 float* final_error,
                                 float* cfc_hidden_state,
                                 size_t hidden_size);
#else
static inline int slam_cfc_direct_vo_get_state(const CfCDirectVO* vo,
                                 int* iterations_used,
                                 float* final_error,
                                 float* cfc_hidden_state,
                                 size_t hidden_size) { (void)vo; (void)iterations_used; (void)final_error; (void)cfc_hidden_state; (void)hidden_size; return -1; }
#endif

// ==================== 稠密地图构建（TSDF） ====================

/**
 * @brief TSDF体素配置结构体
 */
typedef struct {
    float voxel_size;                /**< 体素大小（米），默认0.01（1cm） */
    float truncation_distance;       /**< 截断距离（米），默认=3*voxel_size */
    int volume_dim_x;                /**< 体素网格X方向体素数（默认256） */
    int volume_dim_y;                /**< 体素网格Y方向体素数（默认256） */
    int volume_dim_z;                /**< 体素网格Z方向体素数（默认256） */
    float volume_origin_x;           /**< 体素网格原点X（米） */
    float volume_origin_y;           /**< 体素网格原点Y（米） */
    float volume_origin_z;           /**< 体素网格原点Z（米） */
    int enable_weighted_fusion;      /**< 是否启用加权融合（默认1） */
    float max_weight;                /**< 最大融合权重（默认100） */
    int enable_ray_casting;          /**< 是否启用光线投影渲染（默认1） */
} TSDFVolumeConfig;

/**
 * @brief TSDF体素句柄（不透明）
 *
 * 截断符号距离函数（Truncated Signed Distance Function）体素网格，
 * 用于融合多视角深度图构建稠密3D模型。
 * 算法核心：每帧深度数据通过投影到TSDF体素，更新SDF值和权重，
 * 通过加权平均融合多视角观测。
 */
typedef struct TSDFVolume TSDFVolume;

/**
 * @brief 获取默认TSDF体素配置
 *
 * @return TSDFVolumeConfig 默认配置
 */
#ifdef SELFLNN_SLAM_ADVANCED
TSDFVolumeConfig slam_tsdf_get_default_config(void);
#else
static inline TSDFVolumeConfig slam_tsdf_get_default_config(void) { TSDFVolumeConfig c; memset(&c, 0, sizeof(c)); return c; }
#endif

/**
 * @brief 创建TSDF体素网格
 *
 * @param config TSDF体素配置
 * @return TSDFVolume* 句柄，失败返回NULL
 */
#ifdef SELFLNN_SLAM_ADVANCED
TSDFVolume* slam_tsdf_create(const TSDFVolumeConfig* config);
#else
static inline TSDFVolume* slam_tsdf_create(const TSDFVolumeConfig* config) { (void)config; return NULL; }
#endif

/**
 * @brief 释放TSDF体素网格
 *
 * @param volume TSDF体素句柄
 */
#ifdef SELFLNN_SLAM_ADVANCED
void slam_tsdf_free(TSDFVolume* volume);
#else
static inline void slam_tsdf_free(TSDFVolume* volume) { (void)volume; }
#endif

/**
 * @brief 融合一帧深度图到TSDF体素网格
 *
 * 核心算法：
 * 1. 对深度图中的每个有效像素(u,v)，反投影到3D点P = K^(-1)*[u,v,1]*depth
 * 2. 将P从相机坐标系变换到世界坐标系：Pw = Tcw^(-1)*P
 * 3. 计算Pw对应的体素坐标(vx,vy,vz)
 * 4. 对Pw周围的3x3x3体素邻域，计算SDF值：sdf = ||Pw - voxel_center|| - depth
 * 5. 截断到[-trunc, +trunc]：tsdf = max(-1, min(1, sdf/trunc))
 * 6. 加权融合：tsdf_avg = (w_old*tsdf_old + w_new*tsdf_new)/(w_old + w_new)
 * 7. 权重更新：w = min(w + w_new, max_weight)
 *
 * @param volume TSDF体素句柄
 * @param depth_image 深度图（float数组，值=真实深度米）
 * @param image_width 深度图宽度
 * @param image_height 深度图高度
 * @param camera_params 相机内参 [fx, fy, cx, cy]
 * @param camera_pose 相机位姿 [tx,ty,tz,qw,qx,qy,qz]（世界坐标系）
 * @param max_depth 最大有效深度（米，默认4.0）
 * @param min_depth 最小有效深度（米，默认0.1）
 * @return int 成功返回0，失败返回-1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_tsdf_integrate_depth(TSDFVolume* volume,
                              const float* depth_image,
                              int image_width, int image_height,
                              const float camera_params[4],
                              const float camera_pose[7],
                              float max_depth, float min_depth);
#else
static inline int slam_tsdf_integrate_depth(TSDFVolume* volume,
                              const float* depth_image,
                              int image_width, int image_height,
                              const float camera_params[4],
                              const float camera_pose[7],
                              float max_depth, float min_depth) { (void)volume; (void)depth_image; (void)image_width; (void)image_height; (void)camera_params; (void)camera_pose; (void)max_depth; (void)min_depth; return -1; }
#endif

/**
 * @brief 从TSDF体素网格光线投影渲染深度图和彩色图
 *
 * 核心算法：
 * 1. 对每个输出像素(u,v)，从相机中心沿视线方向步进
 * 2. 在每个步进位置采样TSDF值
 * 3. 检测零交叉面（TSDF值变号位置）
 * 4. 通过线性插值计算精确表面位置
 * 5. 输出渲染深度值和法线方向
 *
 * @param volume TSDF体素句柄
 * @param camera_pose 相机位姿 [tx,ty,tz,qw,qx,qy,qz]
 * @param camera_params 相机内参 [fx, fy, cx, cy]
 * @param output_width 输出图像宽度
 * @param output_height 输出图像高度
 * @param depth_out 输出深度图（float数组，需预分配 output_width*output_height）
 * @param color_out 输出彩色图（float数组，需预分配 output_width*output_height*3，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_tsdf_raycast(const TSDFVolume* volume,
                      const float camera_pose[7],
                      const float camera_params[4],
                      int output_width, int output_height,
                      float* depth_out,
                      float* color_out);
#else
static inline int slam_tsdf_raycast(const TSDFVolume* volume,
                      const float camera_pose[7],
                      const float camera_params[4],
                      int output_width, int output_height,
                      float* depth_out,
                      float* color_out) { (void)volume; (void)camera_pose; (void)camera_params; (void)output_width; (void)output_height; (void)depth_out; (void)color_out; return -1; }
#endif

/**
 * @brief 从TSDF体素网格提取三角网格
 *
 * 使用Marching Cubes算法从TSDF体素网格提取三角网格。
 * 遍历所有体素，对每个体素的8个顶点（立方体）：
 * 1. 根据8个顶点的TSDF符号确定立方体构型（255种可能）
 * 2. 查找边表确定哪些边与零等值面相交
 * 3. 通过线性插值计算交点位置
 * 4. 根据三角表生成三角形
 *
 * @param volume TSDF体素句柄
 * @param vertices 输出顶点数组（每顶点3个float: x,y,z，需预分配，可传NULL查询所需大小）
 * @param normals 输出法线数组（每顶点3个float: nx,ny,nz，可为NULL）
 * @param max_vertices 最大顶点数
 * @param num_vertices 输出实际顶点数
 * @param indices 输出索引数组（每三角形3个int，需预分配）
 * @param max_indices 最大索引数
 * @param num_indices 输出实际索引数
 * @return int 成功返回0，失败返回-1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_tsdf_extract_mesh(const TSDFVolume* volume,
                           float* vertices, float* normals,
                           size_t max_vertices, size_t* num_vertices,
                           int* indices,
                           size_t max_indices, size_t* num_indices);
#else
static inline int slam_tsdf_extract_mesh(const TSDFVolume* volume,
                           float* vertices, float* normals,
                           size_t max_vertices, size_t* num_vertices,
                           int* indices,
                           size_t max_indices, size_t* num_indices) { (void)volume; (void)vertices; (void)normals; (void)max_vertices; (void)num_vertices; (void)indices; (void)max_indices; (void)num_indices; return -1; }
#endif

/**
 * @brief 获取TSDF体素处的SDF值和权重
 *
 * @param volume TSDF体素句柄
 * @param world_x 世界坐标X（米）
 * @param world_y 世界坐标Y（米）
 * @param world_z 世界坐标Z（米）
 * @param sdf_value 输出SDF值
 * @param weight 输出权重（可为NULL）
 * @return int 成功返回0，体素不存在返回1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_tsdf_query_point(const TSDFVolume* volume,
                          float world_x, float world_y, float world_z,
                          float* sdf_value, float* weight);
#else
static inline int slam_tsdf_query_point(const TSDFVolume* volume,
                          float world_x, float world_y, float world_z,
                          float* sdf_value, float* weight) { (void)volume; (void)world_x; (void)world_y; (void)world_z; (void)sdf_value; (void)weight; return 1; }
#endif

/**
 * @brief 重置TSDF体素网格（清空所有数据）
 *
 * @param volume TSDF体素句柄
 */
#ifdef SELFLNN_SLAM_ADVANCED
void slam_tsdf_reset(TSDFVolume* volume);
#else
static inline void slam_tsdf_reset(TSDFVolume* volume) { (void)volume; }
#endif

// ==================== 语义地图构建 ====================

/**
 * @brief 语义地图配置结构体
 */
typedef struct {
    float voxel_size;                /**< 体素大小（米），默认0.02（2cm） */
    int volume_dim_x;                /**< 体素网格X方向体素数 */
    int volume_dim_y;                /**< 体素网格Y方向体素数 */
    int volume_dim_z;                /**< 体素网格Z方向体素数 */
    float volume_origin_x;           /**< 体素网格原点X（米） */
    float volume_origin_y;           /**< 体素网格原点Y（米） */
    float volume_origin_z;           /**< 体素网格原点Z（米） */
    int num_semantic_classes;        /**< 语义类别数（默认20） */
    float min_confidence_threshold;  /**< 最小置信度阈值（默认0.3） */
    float label_smoothing_factor;    /**< 标签平滑因子（默认0.1） */
    int enable_temporal_smoothing;   /**< 是否启用时序平滑（默认1） */
    float temporal_decay_rate;       /**< 时序衰减率（默认0.95） */
} SemanticMapConfig;

/**
 * @brief 语义图句柄（不透明）
 *
 * 语义地图：在体素网格中存储每个体素的语义标签概率分布，
 * 通过多视角语义分割结果融合构建。
 * 每个体素存储一个num_classes维的概率向量，
 * 使用贝叶斯更新融合多帧观测。
 */
typedef struct SemanticMap SemanticMap;

/**
 * @brief 获取默认语义地图配置
 *
 * @return SemanticMapConfig 默认配置
 */
#ifdef SELFLNN_SLAM_ADVANCED
SemanticMapConfig slam_semantic_map_get_default_config(void);
#else
static inline SemanticMapConfig slam_semantic_map_get_default_config(void) { SemanticMapConfig c; memset(&c, 0, sizeof(c)); return c; }
#endif

/**
 * @brief 创建语义地图
 *
 * @param config 语义地图配置
 * @return SemanticMap* 句柄，失败返回NULL
 */
#ifdef SELFLNN_SLAM_ADVANCED
SemanticMap* slam_semantic_map_create(const SemanticMapConfig* config);
#else
static inline SemanticMap* slam_semantic_map_create(const SemanticMapConfig* config) { (void)config; return NULL; }
#endif

/**
 * @brief 释放语义地图
 *
 * @param map 语义地图句柄
 */
#ifdef SELFLNN_SLAM_ADVANCED
void slam_semantic_map_free(SemanticMap* map);
#else
static inline void slam_semantic_map_free(SemanticMap* map) { (void)map; }
#endif

/**
 * @brief 融合一帧语义分割结果到语义地图
 *
 * 核心算法（贝叶斯更新）：
 * 1. 对语义分割图每个像素(u,v)，获取类别概率分布P(c|I(u,v))
 * 2. 通过深度图反投影到3D点Pw
 * 3. 将Pw投影到体素坐标(vx,vy,vz)
 * 4. 对每个影响的体素，执行贝叶斯更新：
 *    P(c|z_1..z_t) ∝ P(z_t|c) * P(c|z_1..z_{t-1})
 *    其中P(z_t|c)为当前帧观测似然（softmax概率）
 * 5. 更新体素的概率分布并归一化
 * 6. 置信度=max(P(c))，标签=argmax(P(c))
 *
 * @param map 语义地图句柄
 * @param semantic_labels 语义标签图（int数组，-1=未知，0~num_classes-1=各类别）
 * @param semantic_probs 语义概率图（float数组，每个像素num_classes个概率值，可为NULL则用one-hot）
 * @param depth_image 深度图（float数组，真实深度米）
 * @param image_width 图像宽度
 * @param image_height 图像高度
 * @param num_classes 语义类别数
 * @param camera_params 相机内参 [fx, fy, cx, cy]
 * @param camera_pose 相机位姿 [tx,ty,tz,qw,qx,qy,qz]
 * @return int 成功返回0，失败返回-1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_semantic_map_integrate(SemanticMap* map,
                                const int* semantic_labels,
                                const float* semantic_probs,
                                const float* depth_image,
                                int image_width, int image_height,
                                int num_classes,
                                const float camera_params[4],
                                const float camera_pose[7]);
#else
static inline int slam_semantic_map_integrate(SemanticMap* map,
                                const int* semantic_labels,
                                const float* semantic_probs,
                                const float* depth_image,
                                int image_width, int image_height,
                                int num_classes,
                                const float camera_params[4],
                                const float camera_pose[7]) { (void)map; (void)semantic_labels; (void)semantic_probs; (void)depth_image; (void)image_width; (void)image_height; (void)num_classes; (void)camera_params; (void)camera_pose; return -1; }
#endif

/**
 * @brief 查询某3D点的语义标签和置信度
 *
 * @param map 语义地图句柄
 * @param world_x 世界坐标X（米）
 * @param world_y 世界坐标Y（米）
 * @param world_z 世界坐标Z（米）
 * @param label 输出语义标签ID
 * @param confidence 输出置信度（0-1）
 * @param probs_out 输出完整概率分布（需预分配num_classes个float，可为NULL）
 * @param max_classes probs_out缓冲区大小
 * @return int 成功返回0，点不存在返回1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_semantic_map_query(const SemanticMap* map,
                            float world_x, float world_y, float world_z,
                            int* label, float* confidence,
                            float* probs_out, int max_classes);
#else
static inline int slam_semantic_map_query(const SemanticMap* map,
                            float world_x, float world_y, float world_z,
                            int* label, float* confidence,
                            float* probs_out, int max_classes) { (void)map; (void)world_x; (void)world_y; (void)world_z; (void)label; (void)confidence; (void)probs_out; (void)max_classes; return 1; }
#endif

/**
 * @brief 重置语义地图
 *
 * @param map 语义地图句柄
 */
#ifdef SELFLNN_SLAM_ADVANCED
void slam_semantic_map_reset(SemanticMap* map);
#else
static inline void slam_semantic_map_reset(SemanticMap* map) { (void)map; }
#endif

/**
 * @brief 获取语义地图统计信息
 *
 * @param map 语义地图句柄
 * @param total_voxels 输出总体素数
 * @param labeled_voxels 输出已标记体素数
 * @param avg_confidence 输出平均置信度
 * @return int 成功返回0，失败返回-1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_semantic_map_get_stats(const SemanticMap* map,
                                int* total_voxels,
                                int* labeled_voxels,
                                float* avg_confidence);
#else
static inline int slam_semantic_map_get_stats(const SemanticMap* map,
                                int* total_voxels,
                                int* labeled_voxels,
                                float* avg_confidence) { (void)map; (void)total_voxels; (void)labeled_voxels; (void)avg_confidence; return -1; }
#endif

// ==================== 动态环境地图更新 ====================

/**
 * @brief 动态地图配置结构体
 */
typedef struct {
    float voxel_size;                /**< 体素大小（米），默认0.02 */
    int volume_dim_x;                /**< 体素网格X方向体素数 */
    int volume_dim_y;                /**< 体素网格Y方向体素数 */
    int volume_dim_z;                /**< 体素网格Z方向体素数 */
    float volume_origin_x;           /**< 体素网格原点X（米） */
    float volume_origin_y;           /**< 体素网格原点Y（米） */
    float volume_origin_z;           /**< 体素网格原点Z（米） */
    float temporal_decay_rate;       /**< 时序衰减率（默认0.98，值越小衰减越快） */
    float change_detection_threshold; /**< 变化检测阈值（默认0.15） */
    float occupancy_decay_constant;  /**< 占用衰减常数（秒，默认10.0） */
    int min_observations_for_stable; /**< 稳定态最小观测次数（默认5） */
    int enable_motion_segmentation;  /**< 是否启用运动分割（默认1） */
    float motion_segmentation_threshold; /**< 运动分割速度阈值（米/帧，默认0.05） */
    int max_tracking_age;            /**< 运动跟踪最大帧龄（默认30帧） */
} DynamicMapConfig;

/**
 * @brief 动态地图句柄（不透明）
 *
 * 动态环境地图：在体素网格中存储占用概率和运动状态，
 * 通过多帧差异检测实现动态物体跟踪和地图更新。
 * 每个体素存储：占用概率、最近更新时间、观测次数、速度估计。
 */
typedef struct DynamicMap DynamicMap;

/**
 * @brief 体素运动状态枚举
 */
typedef enum {
    DYNAMIC_VOXEL_STATE_UNKNOWN = 0,    /**< 未知状态 */
    DYNAMIC_VOXEL_STATE_STATIC = 1,     /**< 静态体素 */
    DYNAMIC_VOXEL_STATE_DYNAMIC = 2,    /**< 动态体素 */
    DYNAMIC_VOXEL_STATE_OCCLUDED = 3,   /**< 被遮挡体素 */
    DYNAMIC_VOXEL_STATE_FREE = 4,       /**< 空闲体素 */
} DynamicVoxelState;

/**
 * @brief 动态体素信息
 */
typedef struct {
    float position[3];               /**< 体素世界坐标（米） */
    float occupancy_prob;            /**< 占用概率（0-1） */
    DynamicVoxelState motion_state;  /**< 运动状态 */
    float velocity[3];               /**< 速度估计（米/帧） */
    float last_update_time;          /**< 最后更新时间戳（秒） */
    int observation_count;           /**< 观测次数 */
    float stability_score;           /**< 稳定度评分（0-1，越高越稳定） */
    int semantic_label;              /**< 语义标签（-1=未知） */
} DynamicVoxelInfo;

/**
 * @brief 获取默认动态地图配置
 *
 * @return DynamicMapConfig 默认配置
 */
#ifdef SELFLNN_SLAM_ADVANCED
DynamicMapConfig slam_dynamic_map_get_default_config(void);
#else
static inline DynamicMapConfig slam_dynamic_map_get_default_config(void) { DynamicMapConfig c; memset(&c, 0, sizeof(c)); return c; }
#endif

/**
 * @brief 创建动态地图
 *
 * @param config 动态地图配置
 * @return DynamicMap* 句柄，失败返回NULL
 */
#ifdef SELFLNN_SLAM_ADVANCED
DynamicMap* slam_dynamic_map_create(const DynamicMapConfig* config);
#else
static inline DynamicMap* slam_dynamic_map_create(const DynamicMapConfig* config) { (void)config; return NULL; }
#endif

/**
 * @brief 释放动态地图
 *
 * @param map 动态地图句柄
 */
#ifdef SELFLNN_SLAM_ADVANCED
void slam_dynamic_map_free(DynamicMap* map);
#else
static inline void slam_dynamic_map_free(DynamicMap* map) { (void)map; }
#endif

/**
 * @brief 更新动态地图（使用当前帧深度/点云）
 *
 * 核心算法：
 * 1. 将当前深度图/点云反投影到世界坐标系
 * 2. 对每个3D点，找到对应的体素
 * 3. 更新体素占用概率（对数几率更新）：
 *    log_odds = log_odds + update - decay*dt
 * 4. 变化检测：对比当前观测与体素历史期望值
 *    如果|当前深度 - 期望深度| > threshold，标记为动态
 * 5. 运动估计：对连续标记为动态的体素，计算速度向量
 * 6. 状态分类：根据速度幅值、持续时间和上下文分类
 * 7. 未被观测到的体素执行指数衰减
 *
 * @param map 动态地图句柄
 * @param depth_image 深度图（float数组，可为NULL如果使用点云）
 * @param point_cloud 点云（float数组，每点3个值xyz，可为NULL如果使用深度图）
 * @param point_count 点云点数（深度图模式下为0）
 * @param image_width 图像宽度（点云模式下为0）
 * @param image_height 图像高度（点云模式下为0）
 * @param camera_params 相机内参 [fx, fy, cx, cy]（深度图模式下使用）
 * @param camera_pose 相机位姿 [tx,ty,tz,qw,qx,qy,qz]
 * @param timestamp 当前时间戳（秒）
 * @return int 成功返回检测到的动态体素数，失败返回-1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_dynamic_map_update(DynamicMap* map,
                            const float* depth_image,
                            const float* point_cloud,
                            size_t point_count,
                            int image_width, int image_height,
                            const float camera_params[4],
                            const float camera_pose[7],
                            float timestamp);
#else
static inline int slam_dynamic_map_update(DynamicMap* map,
                            const float* depth_image,
                            const float* point_cloud,
                            size_t point_count,
                            int image_width, int image_height,
                            const float camera_params[4],
                            const float camera_pose[7],
                            float timestamp) { (void)map; (void)depth_image; (void)point_cloud; (void)point_count; (void)image_width; (void)image_height; (void)camera_params; (void)camera_pose; (void)timestamp; return -1; }
#endif

/**
 * @brief 检测动态变化区域
 *
 * 遍历所有体素，找出状态为DYNAMIC_VOXEL_STATE_DYNAMIC的体素，
 * 并聚类为动态物体区域。
 *
 * @param map 动态地图句柄
 * @param dynamic_voxels 输出动态体素数组（需预分配）
 * @param max_count 最大输出数量
 * @param num_dynamic 输出实际动态体素数
 * @return int 成功返回0，失败返回-1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_dynamic_map_detect_changes(const DynamicMap* map,
                                    DynamicVoxelInfo* dynamic_voxels,
                                    size_t max_count,
                                    size_t* num_dynamic);
#else
static inline int slam_dynamic_map_detect_changes(const DynamicMap* map,
                                    DynamicVoxelInfo* dynamic_voxels,
                                    size_t max_count,
                                    size_t* num_dynamic) { (void)map; (void)dynamic_voxels; (void)max_count; (void)num_dynamic; return -1; }
#endif

/**
 * @brief 查询某3D点的动态状态
 *
 * @param map 动态地图句柄
 * @param world_x 世界坐标X（米）
 * @param world_y 世界坐标Y（米）
 * @param world_z 世界坐标Z（米）
 * @param info 输出动态体素信息（可为NULL）
 * @return int 成功返回0，体素不存在返回1
 */
#ifdef SELFLNN_SLAM_ADVANCED
int slam_dynamic_map_query(const DynamicMap* map,
                           float world_x, float world_y, float world_z,
                           DynamicVoxelInfo* info);
#else
static inline int slam_dynamic_map_query(const DynamicMap* map,
                           float world_x, float world_y, float world_z,
                           DynamicVoxelInfo* info) { (void)map; (void)world_x; (void)world_y; (void)world_z; (void)info; return 1; }
#endif

/**
 * @brief 重置动态地图
 *
 * @param map 动态地图句柄
 */
#ifdef SELFLNN_SLAM_ADVANCED
void slam_dynamic_map_reset(DynamicMap* map);
#else
static inline void slam_dynamic_map_reset(DynamicMap* map) { (void)map; }
#endif

// ==================== CameraInput 相机输入接口 ====================

/**
 * @brief 相机输入源类型枚举
 */
typedef enum {
    CAMERA_SOURCE_SYNTHETIC = 0,  /**< 合成图像（自动生成含视觉特征的测试帧） */
    CAMERA_SOURCE_FILE = 1,       /**< 从文件序列读取（PPM/PGM格式） */
    CAMERA_SOURCE_RAW_VIDEO = 2,  /**< 从原始视频二进制文件读取 */
    CAMERA_SOURCE_HARDWARE = 3,   /**< 从真实相机硬件输入（需通过hardware_interface对接） */
} CameraSourceType;

/**
 * @brief 相机输入帧结构体
 */
typedef struct {
    float* image_data;           /**< 图像数据（float数组，值范围0.0-1.0） */
    int width;                   /**< 图像宽度（像素） */
    int height;                  /**< 图像高度（像素） */
    int channels;                /**< 图像通道数（1=灰度，3=RGB） */
    float timestamp;             /**< 时间戳（秒） */
    int frame_id;                /**< 帧序号（从0开始递增） */
    int is_valid;                /**< 帧数据是否有效 */
} CameraInputFrame;

/**
 * @brief 相机输入配置结构体
 */
typedef struct {
    CameraSourceType source_type; /**< 输入源类型 */
    int image_width;             /**< 图像宽度（像素） */
    int image_height;            /**< 图像高度（像素） */
    int channels;                /**< 图像通道数（默认1=灰度） */
    float camera_params[4];      /**< 相机内参 [fx, fy, cx, cy] */
    
    /* 合成源参数 */
    int synthetic_total_frames;  /**< 合成序列总帧数（默认100） */
    
    /* 文件源参数 */
    const char* file_directory;  /**< 图像文件目录（CAMERA_SOURCE_FILE时有效） */
    const char* file_prefix;     /**< 文件名前缀（如"frame_"） */
    const char* file_extension;  /**< 文件扩展名（如"ppm"、"pgm"） */
    int file_start_index;        /**< 起始文件编号 */
    
    /* 原始视频源参数 */
    const char* raw_video_path;  /**< 原始视频文件路径（CAMERA_SOURCE_RAW_VIDEO时有效） */
    int raw_video_total_frames;  /**< 原始视频总帧数（0表示自动推算） */
} CameraInputConfig;

/**
 * @brief 相机输入句柄（不透明）
 *
 * 统一的相机输入抽象层，支持四种输入源：
 * - CAMERA_SOURCE_SYNTHETIC：自动生成含视觉特征的合成帧
 * - CAMERA_SOURCE_FILE：从磁盘按编号顺序读取PPM/PGM图像文件
 * - CAMERA_SOURCE_RAW_VIDEO：从原始二进制视频文件读取帧
 * - CAMERA_SOURCE_HARDWARE：通过硬件接口连接真实相机
 *
 * 内部自动管理对应的底层资源（帧读取器、视频读取器等），
 * 对外提供统一的帧读取接口。
 */
typedef struct CameraInput CameraInput;

/**
 * @brief 获取默认相机输入配置
 *
 * @return CameraInputConfig 默认配置（CAMERA_SOURCE_SYNTHETIC，640x480灰度）
 */
CameraInputConfig camera_input_get_default_config(void);

/**
 * @brief 创建相机输入接口
 *
 * 根据配置中的source_type自动创建对应的底层输入源。
 * 支持合成、文件、原始视频、硬件四种模式。
 *
 * @param config 相机输入配置
 * @return CameraInput* 相机输入句柄，失败返回NULL
 */
CameraInput* camera_input_create(const CameraInputConfig* config);

/**
 * @brief 读取下一帧图像
 *
 * 从当前相机输入源读取下一帧图像。
 * 内部自动管理帧缓冲区的分配和释放。
 *
 * @param camera 相机输入句柄
 * @param frame 帧输出结构体（image_data指向内部缓冲区，调用者无需释放）
 * @return int 成功返回1（还有更多帧），返回0（已到末尾），失败返回-1
 */
int camera_input_read_frame(CameraInput* camera, CameraInputFrame* frame);

/**
 * @brief 重置相机输入到起始位置
 *
 * @param camera 相机输入句柄
 * @return int 成功返回0，失败返回-1
 */
int camera_input_reset(CameraInput* camera);

/**
 * @brief 获取相机输入信息
 *
 * @param camera 相机输入句柄
 * @param source_type 输入源类型输出
 * @param total_frames 总帧数输出（合成源和原始视频源有效，文件源-1表示未知）
 * @param current_frame 当前帧序号输出
 * @return int 成功返回0，失败返回-1
 */
int camera_input_get_info(const CameraInput* camera,
                         int* source_type,
                         int* total_frames,
                         int* current_frame);

/**
 * @brief 释放相机输入接口
 *
 * 自动释放内部管理的所有底层资源。
 *
 * @param camera 相机输入句柄
 */
void camera_input_free(CameraInput* camera);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_SLAM_H