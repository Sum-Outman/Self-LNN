/**
 * @file point_cloud.h
 * @brief 点云处理管道接口
 * 
 * 点云处理模块，支持点云滤波、分割、配准、特征提取等操作。
 * 实现完整的点云处理管道，用于3D感知和环境建模。
 */

#ifndef SELFLNN_POINT_CLOUD_H
#define SELFLNN_POINT_CLOUD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 点云滤波方法枚举
 */
typedef enum {
    POINT_CLOUD_FILTER_VOXEL_GRID = 0,     /**< 体素网格滤波 */
    POINT_CLOUD_FILTER_STATISTICAL = 1,    /**< 统计滤波 */
    POINT_CLOUD_FILTER_RADIUS = 2,         /**< 半径滤波 */
    POINT_CLOUD_FILTER_CONDITIONAL = 3,    /**< 条件滤波 */
    POINT_CLOUD_FILTER_PASSTHROUGH = 4,    /**< 直通滤波 */
} PointCloudFilterMethod;

/**
 * @brief 点云分割方法枚举
 */
typedef enum {
    POINT_CLOUD_SEGMENTATION_PLANE = 0,    /**< 平面分割（RANSAC） */
    POINT_CLOUD_SEGMENTATION_EUCLIDEAN = 1,/**< 欧几里得聚类分割 */
    POINT_CLOUD_SEGMENTATION_REGION = 2,   /**< 区域生长分割 */
    POINT_CLOUD_SEGMENTATION_COLOR = 3,    /**< 颜色分割 */
    POINT_CLOUD_SEGMENTATION_NORMAL = 4,   /**< 法线分割 */
} PointCloudSegmentationMethod;

/**
 * @brief 点云配准方法枚举
 */
typedef enum {
    POINT_CLOUD_REGISTRATION_ICP = 0,      /**< 迭代最近点算法 */
    POINT_CLOUD_REGISTRATION_NDT = 1,      /**< 正态分布变换 */
    POINT_CLOUD_REGISTRATION_FPFH = 2,     /**< 基于FPFH特征的配准 */
    POINT_CLOUD_REGISTRATION_SAC = 3,      /**< 采样一致性初始配准 */
    POINT_CLOUD_REGISTRATION_COLOR = 4,    /**< 彩色点云配准 */
} PointCloudRegistrationMethod;

/**
 * @brief 点云特征类型枚举
 */
typedef enum {
    POINT_CLOUD_FEATURE_NORMALS = 0,       /**< 法线特征 */
    POINT_CLOUD_FEATURE_FPFH = 1,          /**< 快速点特征直方图 */
    POINT_CLOUD_FEATURE_SHOT = 2,          /**< SHOT特征 */
    POINT_CLOUD_FEATURE_ISS = 3,           /**< 内部形状描述子 */
    POINT_CLOUD_FEATURE_GRSD = 4,          /**< 全局半径表面描述子 */
} PointCloudFeatureType;

/**
 * @brief 点云数据结构体
 */
typedef struct {
    float* points;          /**< 点云数据（XYZ坐标，每点3个浮点数） */
    float* colors;          /**< 颜色数据（RGB，每点3个浮点数，可选） */
    float* normals;         /**< 法线数据（每点3个浮点数，可选） */
    float* intensities;     /**< 强度数据（每点1个浮点数，可选） */
    size_t num_points;      /**< 点数 */
    int has_colors;         /**< 是否有颜色数据 */
    int has_normals;        /**< 是否有法线数据 */
    int has_intensities;    /**< 是否有强度数据 */
    float min_bounds[3];    /**< 最小边界（x, y, z） */
    float max_bounds[3];    /**< 最大边界（x, y, z） */
    float centroid[3];      /**< 质心 */
    float covariance[9];    /**< 协方差矩阵（3x3，行主序） */
} PointCloud;

/**
 * @brief 点云滤波配置结构体
 */
typedef struct {
    PointCloudFilterMethod method;         /**< 滤波方法 */
    float voxel_size;                      /**< 体素大小（米，用于体素网格滤波） */
    int mean_k;                            /**< 邻居点数（用于统计滤波） */
    float stddev_mult;                     /**< 标准差乘数（用于统计滤波） */
    float radius;                          /**< 搜索半径（米，用于半径滤波） */
    int min_neighbors;                     /**< 最小邻居数（用于半径滤波） */
    float min_limit;                       /**< 最小值限制（用于直通滤波） */
    float max_limit;                       /**< 最大值限制（用于直通滤波） */
    int filter_field;                      /**< 滤波字段：0=x, 1=y, 2=z, 3=intensity */
    int keep_organized;                    /**< 是否保持有序结构 */
} PointCloudFilterConfig;

/**
 * @brief 点云分割配置结构体
 */
typedef struct {
    PointCloudSegmentationMethod method;   /**< 分割方法 */
    float distance_threshold;              /**< 距离阈值（米，用于平面分割） */
    int max_iterations;                    /**< 最大迭代次数（用于RANSAC） */
    float cluster_tolerance;               /**< 聚类容差（米，用于欧几里得聚类） */
    int min_cluster_size;                  /**< 最小聚类点数 */
    int max_cluster_size;                  /**< 最大聚类点数 */
    float curvature_threshold;             /**< 曲率阈值（用于区域生长） */
    float smoothness_threshold;            /**< 平滑度阈值（用于区域生长） */
    int normal_neighbors;                  /**< 法线计算邻居数 */
    float color_threshold;                 /**< 颜色阈值（用于颜色分割） */
} PointCloudSegmentationConfig;

/**
 * @brief 点云配准配置结构体
 */
typedef struct {
    PointCloudRegistrationMethod method;   /**< 配准方法 */
    float max_correspondence_distance;     /**< 最大对应点距离（米） */
    int max_iterations;                    /**< 最大迭代次数 */
    float transformation_epsilon;          /**< 变换收敛阈值 */
    float fitness_epsilon;                 /**< 拟合度收敛阈值 */
    int use_reciprocal_correspondences;    /**< 是否使用互相对应关系 */
    float max_correspondence_distance_final; /**< 最终配准的最大对应点距离 */
    float inlier_threshold;                /**< 内点阈值 */
    int num_samples;                       /**< 采样点数 */
    int correspondence_randomness;         /**< 对应点随机性 */
    float similarity_threshold;            /**< 相似度阈值 */
} PointCloudRegistrationConfig;

/**
 * @brief 点云特征提取配置结构体
 */
typedef struct {
    PointCloudFeatureType feature_type;    /**< 特征类型 */
    float normal_search_radius;            /**< 法线搜索半径（米） */
    float feature_search_radius;           /**< 特征搜索半径（米） */
    int normal_neighbors;                  /**< 法线计算邻居数 */
    int feature_neighbors;                 /**< 特征计算邻居数 */
    int compute_normals;                   /**< 是否计算法线 */
    int compute_features;                  /**< 是否计算特征 */
    int use_gpu;                           /**< 是否使用GPU加速 */
} PointCloudFeatureConfig;

/**
 * @brief 点云分割结果结构体
 */
typedef struct {
    PointCloud* clusters;                  /**< 分割出的聚类数组 */
    int num_clusters;                      /**< 聚类数量 */
    int* cluster_indices;                  /**< 每个点的聚类索引 */
    float* plane_coefficients;             /**< 平面系数（如果分割平面） */
    int plane_inliers;                     /**< 平面内点数 */
    float segmentation_time_ms;            /**< 分割时间（毫秒） */
} PointCloudSegmentationResult;

/**
 * @brief 点云配准结果结构体
 */
typedef struct {
    float transformation[16];              /**< 变换矩阵（4x4，行主序） */
    float fitness_score;                   /**< 配准拟合度分数 */
    float inlier_rmse;                     /**< 内点均方根误差 */
    int num_iterations;                    /**< 实际迭代次数 */
    float registration_time_ms;            /**< 配准时间（毫秒） */
    int convergence;                       /**< 是否收敛 */
} PointCloudRegistrationResult;

/**
 * @brief 点云特征提取结果结构体
 */
typedef struct {
    float* normals;                        /**< 法线数组（每点3个浮点数） */
    float* features;                       /**< 特征数组（每点特征维度） */
    int feature_dimension;                 /**< 特征维度 */
    size_t num_points_with_features;       /**< 有特征的点数 */
    float feature_extraction_time_ms;      /**< 特征提取时间（毫秒） */
} PointCloudFeatureResult;

/**
 * @brief 点云处理器句柄
 */
typedef struct PointCloudProcessor PointCloudProcessor;

/**
 * @brief 创建点云处理器
 * 
 * @return PointCloudProcessor* 处理器句柄，失败返回NULL
 */
PointCloudProcessor* point_cloud_processor_create();

/**
 * @brief 释放点云处理器
 * 
 * @param processor 处理器句柄
 */
void point_cloud_processor_free(PointCloudProcessor* processor);

/**
 * @brief 从深度图创建点云
 * 
 * @param processor 处理器句柄
 * @param depth_map 深度图数据
 * @param width 深度图宽度
 * @param height 深度图高度
 * @param calibration 相机标定参数
 * @param point_cloud 点云输出
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_from_depth(PointCloudProcessor* processor,
                          const float* depth_map, int width, int height,
                          const void* calibration,  /* 应为CameraCalibration类型 */
                          PointCloud* point_cloud);

/**
 * @brief 滤波点云
 * 
 * @param processor 处理器句柄
 * @param input 输入点云
 * @param config 滤波配置
 * @param output 滤波后点云输出
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_filter(PointCloudProcessor* processor,
                      const PointCloud* input,
                      const PointCloudFilterConfig* config,
                      PointCloud* output);

/**
 * @brief 分割点云
 * 
 * @param processor 处理器句柄
 * @param input 输入点云
 * @param config 分割配置
 * @param result 分割结果输出
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_segment(PointCloudProcessor* processor,
                       const PointCloud* input,
                       const PointCloudSegmentationConfig* config,
                       PointCloudSegmentationResult* result);

/**
 * @brief 配准点云（点云对齐）
 * 
 * @param processor 处理器句柄
 * @param source 源点云
 * @param target 目标点云
 * @param config 配准配置
 * @param result 配准结果输出
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_register(PointCloudProcessor* processor,
                        const PointCloud* source,
                        const PointCloud* target,
                        const PointCloudRegistrationConfig* config,
                        PointCloudRegistrationResult* result);

/**
 * @brief 提取点云特征
 * 
 * @param processor 处理器句柄
 * @param input 输入点云
 * @param config 特征提取配置
 * @param result 特征提取结果输出
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_extract_features(PointCloudProcessor* processor,
                                const PointCloud* input,
                                const PointCloudFeatureConfig* config,
                                PointCloudFeatureResult* result);

/**
 * @brief 计算点云法线
 * 
 * @param processor 处理器句柄
 * @param input 输入点云
 * @param search_radius 搜索半径（米）
 * @param max_neighbors 最大邻居数
 * @param normals 法线输出数组（每点3个浮点数）
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_compute_normals(PointCloudProcessor* processor,
                               const PointCloud* input,
                               float search_radius, int max_neighbors,
                               float* normals);

/**
 * @brief 下采样点云
 * 
 * @param processor 处理器句柄
 * @param input 输入点云
 * @param voxel_size 体素大小（米）
 * @param output 下采样后点云输出
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_downsample(PointCloudProcessor* processor,
                          const PointCloud* input, float voxel_size,
                          PointCloud* output);

/**
 * @brief 移除离群点
 * 
 * @param processor 处理器句柄
 * @param input 输入点云
 * @param mean_k 邻居点数
 * @param stddev_mult 标准差乘数
 * @param output 滤波后点云输出
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_remove_outliers(PointCloudProcessor* processor,
                               const PointCloud* input,
                               int mean_k, float stddev_mult,
                               PointCloud* output);

/**
 * @brief 估计点云平面
 * 
 * @param processor 处理器句柄
 * @param input 输入点云
 * @param distance_threshold 距离阈值（米）
 * @param max_iterations 最大迭代次数
 * @param plane_coefficients 平面系数输出（4个浮点数：a,b,c,d for ax+by+cz+d=0）
 * @param inlier_indices 内点索引输出数组
 * @param max_inliers 最大内点数
 * @return int 成功返回内点数，失败返回-1
 */
int point_cloud_estimate_plane(PointCloudProcessor* processor,
                              const PointCloud* input,
                              float distance_threshold, int max_iterations,
                              float* plane_coefficients,
                              int* inlier_indices, int max_inliers);

/**
 * @brief 计算点云边界框
 * 
 * @param point_cloud 点云
 * @param min_bounds 最小边界输出（3个浮点数）
 * @param max_bounds 最大边界输出（3个浮点数）
 * @param centroid 质心输出（3个浮点数）
 */
void point_cloud_compute_bounds(const PointCloud* point_cloud,
                               float* min_bounds, float* max_bounds,
                               float* centroid);

/**
 * @brief 计算点云协方差矩阵
 * 
 * @param point_cloud 点云
 * @param covariance 协方差矩阵输出（9个浮点数，3x3行主序）
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_compute_covariance(const PointCloud* point_cloud,
                                  float* covariance);

/**
 * @brief 保存点云到文件
 * 
 * @param point_cloud 点云
 * @param filepath 文件路径
 * @param format 文件格式：0=PLY, 1=PCD, 2=OBJ, 3=CSV
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_save_to_file(const PointCloud* point_cloud,
                            const char* filepath, int format);

/**
 * @brief 从文件加载点云
 * 
 * @param point_cloud 点云输出
 * @param filepath 文件路径
 * @param format 文件格式：0=PLY, 1=PCD, 2=OBJ, 3=CSV
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_load_from_file(PointCloud* point_cloud,
                              const char* filepath, int format);

/**
 * @brief 合并多个点云
 * 
 * @param processor 处理器句柄
 * @param point_clouds 点云数组
 * @param num_clouds 点云数量
 * @param transform 变换矩阵数组（可为NULL）
 * @param merged 合并后点云输出
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_merge(PointCloudProcessor* processor,
                     const PointCloud* point_clouds, int num_clouds,
                     const float* transforms,  /* 4x4矩阵数组 */
                     PointCloud* merged);

/**
 * @brief 计算点云密度
 * 
 * @param point_cloud 点云
 * @param search_radius 搜索半径（米）
 * @return float 平均点密度（点数/立方米）
 */
float point_cloud_compute_density(const PointCloud* point_cloud,
                                 float search_radius);

/**
 * @brief 创建空点云
 * 
 * @param point_cloud 点云输出
 * @param capacity 初始容量
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_create_empty(PointCloud* point_cloud, size_t capacity);

/**
 * @brief 释放点云内存
 * 
 * @param point_cloud 点云
 */
void point_cloud_free(PointCloud* point_cloud);

/**
 * @brief 复制点云
 * 
 * @param dest 目标点云
 * @param src 源点云
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_copy(PointCloud* dest, const PointCloud* src);

/**
 * @brief 添加点到点云
 * 
 * @param point_cloud 点云
 * @param x X坐标
 * @param y Y坐标
 * @param z Z坐标
 * @param r 红色（0-1，可选）
 * @param g 绿色（0-1，可选）
 * @param b 蓝色（0-1，可选）
 * @param intensity 强度（可选）
 * @return int 成功返回0，失败返回-1
 */
int point_cloud_add_point(PointCloud* point_cloud,
                         float x, float y, float z,
                         float r, float g, float b,
                         float intensity);

/**
 * @brief 重置点云处理器
 * 
 * @param processor 处理器句柄
 */
void point_cloud_processor_reset(PointCloudProcessor* processor);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_POINT_CLOUD_H