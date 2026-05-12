/**
 * @file stereo_calibration.h
 * @brief 立体视觉相机标定接口
 * 
 * 双目相机标定模块，支持相机内参、外参、畸变系数标定，
 * 以及立体校正和立体校正映射计算。
 */

#ifndef SELFLNN_STEREO_CALIBRATION_H
#define SELFLNN_STEREO_CALIBRATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 相机内参结构体
 */
typedef struct {
    float fx;           /**< 焦距x（像素单位） */
    float fy;           /**< 焦距y（像素单位） */
    float cx;           /**< 主点x（像素单位） */
    float cy;           /**< 主点y（像素单位） */
    float k1;           /**< 径向畸变系数1 */
    float k2;           /**< 径向畸变系数2 */
    float k3;           /**< 径向畸变系数3 */
    float p1;           /**< 切向畸变系数1 */
    float p2;           /**< 切向畸变系数2 */
    int image_width;    /**< 图像宽度 */
    int image_height;   /**< 图像高度 */
} CameraIntrinsics;

/**
 * @brief 相机外参结构体
 */
typedef struct {
    float rotation[9];      /**< 旋转矩阵（3x3，行主序） */
    float translation[3];   /**< 平移向量（米） */
} CameraExtrinsics;

/**
 * @brief 立体相机标定参数结构体
 */
typedef struct {
    CameraIntrinsics left_intrinsics;   /**< 左相机内参 */
    CameraIntrinsics right_intrinsics;  /**< 右相机内参 */
    CameraExtrinsics extrinsics;        /**< 右相机相对于左相机的外参 */
    float essential_matrix[9];          /**< 本质矩阵 */
    float fundamental_matrix[9];        /**< 基础矩阵 */
    float rectification_left[9];        /**< 左相机校正矩阵 */
    float rectification_right[9];       /**< 右相机校正矩阵 */
    float projection_left[12];          /**< 左相机投影矩阵（3x4） */
    float projection_right[12];         /**< 右相机投影矩阵（3x4） */
    float disparity_to_depth[16];       /**< 视差转深度矩阵（4x4） */
    float baseline;                     /**< 基线长度（米） */
    int is_calibrated;                  /**< 是否已标定 */
} StereoCalibration;

/**
 * @brief 标定板类型枚举
 */
typedef enum {
    CALIBRATION_PATTERN_CHESSBOARD = 0,     /**< 棋盘格 */
    CALIBRATION_PATTERN_CIRCLES_GRID = 1,   /**< 圆形网格 */
    CALIBRATION_PATTERN_ASYMMETRIC_CIRCLES = 2, /**< 非对称圆形网格 */
} CalibrationPattern;

/**
 * @brief 相机标定配置结构体
 */
typedef struct {
    CalibrationPattern pattern_type;    /**< 标定板类型 */
    int pattern_width;                  /**< 标定板角点宽度（内角点数） */
    int pattern_height;                 /**< 标定板角点高度（内角点数） */
    float square_size;                  /**< 标定板方格大小（米） */
    int max_iterations;                 /**< 最大迭代次数 */
    float accuracy_epsilon;             /**< 精度收敛阈值 */
    int use_intrinsic_guess;            /**< 是否使用内参初始猜测 */
    int fix_principal_point;            /**< 是否固定主点 */
    int fix_aspect_ratio;               /**< 是否固定纵横比 */
    int zero_tangential_distortion;     /**< 是否强制切向畸变为零 */
    int use_rational_model;             /**< 是否使用有理畸变模型 */
} CalibrationConfig;

/**
 * @brief 相机标定结果结构体
 */
typedef struct {
    CameraIntrinsics intrinsics;        /**< 相机内参 */
    float reprojection_error;           /**< 重投影误差（像素） */
    float distortion_error;             /**< 畸变误差 */
    int calibration_valid;              /**< 标定是否有效 */
    float calibration_time_ms;          /**< 标定时间（毫秒） */
} CalibrationResult;

/**
 * @brief 立体标定结果结构体
 */
typedef struct {
    StereoCalibration calibration;      /**< 立体标定参数 */
    float reprojection_error;           /**< 立体重投影误差 */
    float epipolar_error;               /**< 极线误差 */
    int calibration_valid;              /**< 标定是否有效 */
    float calibration_time_ms;          /**< 标定时间（毫秒） */
} StereoCalibrationResult;

/**
 * @brief 立体校正映射结构体
 */
typedef struct {
    float* map_left_x;                  /**< 左相机x方向校正映射 */
    float* map_left_y;                  /**< 左相机y方向校正映射 */
    float* map_right_x;                 /**< 右相机x方向校正映射 */
    float* map_right_y;                 /**< 右相机y方向校正映射 */
    int map_width;                      /**< 映射宽度 */
    int map_height;                     /**< 映射高度 */
} StereoRectificationMap;

/**
 * @brief 相机标定器句柄
 */
typedef struct CameraCalibrator CameraCalibrator;

/**
 * @brief 创建相机标定器
 * 
 * @param config 标定配置
 * @return CameraCalibrator* 标定器句柄，失败返回NULL
 */
CameraCalibrator* camera_calibrator_create(const CalibrationConfig* config);

/**
 * @brief 释放相机标定器
 * 
 * @param calibrator 标定器句柄
 */
void camera_calibrator_free(CameraCalibrator* calibrator);

/**
 * @brief 标定单目相机
 * 
 * @param calibrator 标定器句柄
 * @param images 标定图像数组
 * @param num_images 图像数量
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param result 标定结果输出
 * @return int 成功返回0，失败返回-1
 */
int camera_calibrate_monocular(CameraCalibrator* calibrator,
                              const float** images, int num_images,
                              int width, int height, int channels,
                              CalibrationResult* result);

/**
 * @brief 标定立体相机
 * 
 * @param calibrator 标定器句柄
 * @param left_images 左相机图像数组
 * @param right_images 右相机图像数组
 * @param num_images 图像数量
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param result 立体标定结果输出
 * @return int 成功返回0，失败返回-1
 */
int camera_calibrate_stereo(CameraCalibrator* calibrator,
                           const float** left_images, const float** right_images,
                           int num_images, int width, int height, int channels,
                           StereoCalibrationResult* result);

/**
 * @brief 保存相机标定参数到文件
 * 
 * @param intrinsics 相机内参
 * @param extrinsics 相机外参（可选）
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int camera_calibration_save(const CameraIntrinsics* intrinsics,
                           const CameraExtrinsics* extrinsics,
                           const char* filepath);

/**
 * @brief 从文件加载相机标定参数
 * 
 * @param intrinsics 相机内参输出
 * @param extrinsics 相机外参输出（可选）
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int camera_calibration_load(CameraIntrinsics* intrinsics,
                           CameraExtrinsics* extrinsics,
                           const char* filepath);

/**
 * @brief 保存立体标定参数到文件
 * 
 * @param calibration 立体标定参数
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int stereo_calibration_save(const StereoCalibration* calibration,
                           const char* filepath);

/**
 * @brief 从文件加载立体标定参数
 * 
 * @param calibration 立体标定参数输出
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int stereo_calibration_load(StereoCalibration* calibration,
                           const char* filepath);

/**
 * @brief 计算立体校正映射
 * 
 * @param calibration 立体标定参数
 * @param width 图像宽度
 * @param height 图像高度
 * @param rectification_map 校正映射输出
 * @return int 成功返回0，失败返回-1
 */
int stereo_compute_rectification_map(const StereoCalibration* calibration,
                                    int width, int height,
                                    StereoRectificationMap* rectification_map);

/**
 * @brief 释放立体校正映射内存
 * 
 * @param map 校正映射
 */
void stereo_rectification_map_free(StereoRectificationMap* map);

/**
 * @brief 应用立体校正
 * 
 * @param src_image 源图像
 * @param dst_image 目标图像输出
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param map_x x方向映射
 * @param map_y y方向映射
 * @return int 成功返回0，失败返回-1
 */
int stereo_apply_rectification(const float* src_image, float* dst_image,
                              int width, int height, int channels,
                              const float* map_x, const float* map_y);

/**
 * @brief 计算视差图到深度图的转换矩阵
 * 
 * @param calibration 立体标定参数
 * @param disparity_to_depth 转换矩阵输出（4x4，行主序）
 * @return int 成功返回0，失败返回-1
 */
int stereo_compute_disparity_to_depth(const StereoCalibration* calibration,
                                     float* disparity_to_depth);

/**
 * @brief 将视差图转换为深度图
 * 
 * @param disparity_map 视差图数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param calibration 立体标定参数
 * @param depth_map 深度图输出
 * @return int 成功返回0，失败返回-1
 */
int stereo_disparity_to_depth(const float* disparity_map, int width, int height,
                             const StereoCalibration* calibration,
                             float* depth_map);

/**
 * @brief 计算极线误差
 * 
 * @param calibration 立体标定参数
 * @param left_points 左相机点集（2D）
 * @param right_points 右相机点集（2D）
 * @param num_points 点数
 * @param epipolar_errors 极线误差输出
 * @return float 平均极线误差
 */
float stereo_compute_epipolar_error(const StereoCalibration* calibration,
                                   const float* left_points,
                                   const float* right_points,
                                   int num_points, float* epipolar_errors);

/**
 * @brief 验证标定结果
 * 
 * @param calibration 立体标定参数
 * @param test_images 测试图像数组
 * @param num_test_images 测试图像数量
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param reprojection_errors 重投影误差输出
 * @return float 平均重投影误差
 */
float stereo_validate_calibration(const StereoCalibration* calibration,
                                 const float** test_images,
                                 int num_test_images,
                                 int width, int height, int channels,
                                 float* reprojection_errors);

/**
 * @brief 获取默认标定配置
 * 
 * @param config 标定配置输出
 */
void camera_calibration_default_config(CalibrationConfig* config);

/**
 * @brief 获取默认相机内参
 * 
 * @param intrinsics 相机内参输出
 * @param width 图像宽度
 * @param height 图像高度
 */
void camera_calibration_default_intrinsics(CameraIntrinsics* intrinsics,
                                          int width, int height);

/**
 * @brief 重置相机标定器
 * 
 * @param calibrator 标定器句柄
 */
void camera_calibrator_reset(CameraCalibrator* calibrator);

/**
 * @brief 检测标定板角点
 * 
 * @param image 输入图像
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param pattern_type 标定板类型
 * @param pattern_width 角点宽度
 * @param pattern_height 角点高度
 * @param corners 角点坐标输出（2D）
 * @param max_corners 最大角点数
 * @return int 成功返回检测到的角点数，失败返回-1
 */
int camera_calibration_detect_corners(const float* image,
                                     int width, int height, int channels,
                                     CalibrationPattern pattern_type,
                                     int pattern_width, int pattern_height,
                                     float* corners, int max_corners);

/**
 * @brief 计算标定板角点的3D世界坐标
 * 
 * @param pattern_width 角点宽度
 * @param pattern_height 角点高度
 * @param square_size 方格大小
 * @param object_points 3D坐标输出
 * @param max_points 最大点数
 * @return int 成功返回角点数，失败返回-1
 */
int camera_calibration_compute_object_points(int pattern_width, int pattern_height,
                                            float square_size,
                                            float* object_points, int max_points);

/**
 * @brief 估计相机姿态（PnP问题）
 * 
 * @param object_points 3D物体点
 * @param image_points 2D图像点
 * @param num_points 点数
 * @param intrinsics 相机内参
 * @param rotation 旋转向量输出（3个浮点数）
 * @param translation 平移向量输出（3个浮点数）
 * @return int 成功返回0，失败返回-1
 */
int camera_calibration_estimate_pose(const float* object_points,
                                    const float* image_points, int num_points,
                                    const CameraIntrinsics* intrinsics,
                                    float* rotation, float* translation);

/* ============================================================================
 * 非线性优化标定（Bundle Adjustment）
 *
 * 使用 Levenberg-Marquardt 算法对标定参数进行非线性优化，
 * 最小化所有观测点的重投影误差。
 * ============================================================================ */

/**
 * @brief Bundle Adjustment 配置结构体
 */
typedef struct {
    int max_iterations;             /**< 最大迭代次数（默认100） */
    float lambda_init;              /**< LM算法初始阻尼系数（默认0.001） */
    float lambda_factor;            /**< LM阻尼调整因子（默认10.0） */
    float gradient_threshold;       /**< 梯度收敛阈值（默认1e-6） */
    float error_threshold;          /**< 误差变化收敛阈值（默认1e-8） */
    int fix_intrinsics;             /**< 是否固定内参（仅优化外参） */
    int fix_distortion;             /**< 是否固定畸变系数 */
    int verbose;                    /**< 是否输出优化过程信息 */
} BundleAdjustmentConfig;

/**
 * @brief 获取默认 Bundle Adjustment 配置
 *
 * @param config 配置输出
 */
void bundle_adjustment_default_config(BundleAdjustmentConfig* config);

/**
 * @brief 执行双目 Bundle Adjustment 非线性优化
 *
 * 使用 Levenberg-Marquardt 算法联合优化左右相机内参、畸变系数和外参，
 * 最小化所有标定图像角点的重投影误差。
 *
 * @param calibration [in/out] 待优化的立体标定参数
 * @param left_image_points 所有左图像角点数组 [num_images * corners_per_image * 2]
 * @param right_image_points 所有右图像角点数组 [num_images * corners_per_image * 2]
 * @param object_points 3D世界坐标数组 [corners_per_image * 3]
 * @param num_images 标定图像对数
 * @param corners_per_image 每幅图像的角点数
 * @param config BA配置
 * @param final_error [out] 优化后的平均重投影误差
 * @return int 成功返回0，失败返回-1
 */
int stereo_bundle_adjustment(StereoCalibration* calibration,
                             const float* left_image_points,
                             const float* right_image_points,
                             const float* object_points,
                             int num_images,
                             int corners_per_image,
                             const BundleAdjustmentConfig* config,
                             float* final_error);

/**
 * @brief 执行单目 Bundle Adjustment
 *
 * @param intrinsics [in/out] 相机内参（含畸变）
 * @param image_points 所有图像角点数组 [num_images * corners_per_image * 2]
 * @param object_points 3D世界坐标数组 [corners_per_image * 3]
 * @param num_images 图像数量
 * @param corners_per_image 每幅图像角点数
 * @param rotations [out] 每幅图像的旋转向量 [num_images * 3]
 * @param translations [out] 每幅图像的平移向量 [num_images * 3]
 * @param config BA配置
 * @param final_error [out] 优化后的平均重投影误差
 * @return int 成功返回0，失败返回-1
 */
int monocular_bundle_adjustment(CameraIntrinsics* intrinsics,
                                const float* image_points,
                                const float* object_points,
                                int num_images,
                                int corners_per_image,
                                float* rotations,
                                float* translations,
                                const BundleAdjustmentConfig* config,
                                float* final_error);

/**
 * @brief 计算 3D 点云（从视差图 + 标定参数）
 *
 * 将视差图转换为 3D 点云，生成 (x,y,z) 坐标数组。
 * 用于前端 3D 实时重建显示。
 *
 * @param disparity_map 视差图数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param calibration 立体标定参数
 * @param point_cloud 点云输出 [width * height * 3]
 * @param max_valid_points 最大有效点数
 * @return int 实际有效点数，失败返回-1
 */
int stereo_compute_point_cloud(const float* disparity_map,
                               int width, int height,
                               const StereoCalibration* calibration,
                               float* point_cloud, int max_valid_points);

/* ============================================================================
 * 双目立体匹配（Block Matching / SAD 算法）
 *
 * 从校正后的左右图像计算视差图。
 * 这是"双目空间识别"的核心算法，实现完整的：
 *   左目图像 + 右目图像 → Block Matching → 视差图 → 深度图 → 3D点云
 * ============================================================================ */

/**
 * @brief 双目立体匹配：使用块匹配(SAD)计算视差图
 *
 * 在右图中搜索左图每个像素的对应点，使用绝对差和(SAD)作为匹配代价。
 * 窗口大小和视差范围可配置。这是经典的双目立体视觉基础算法。
 *
 * @param left_img 左目图像（已校正，float灰度或RGB，按channel步进）
 * @param right_img 右目图像（已校正）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数（1=灰度，3=RGB）
 * @param disparity_map 输出视差图 [width * height]，视差值×16（亚像素精度）
 * @param min_disparity 最小视差（像素），默认0
 * @param max_disparity 最大视差（像素），默认64
 * @param window_size 匹配窗口大小，默认9（奇数）
 * @return 0成功，-1失败
 */
int stereo_compute_disparity(const float* left_img, const float* right_img,
                             int width, int height, int channels,
                             float* disparity_map,
                             int min_disparity, int max_disparity,
                             int window_size);

/**
 * @brief 将视差图转换为深度图
 *
 * depth = baseline * fx / (disparity + cx_diff)
 *
 * @param disparity_map 视差图
 * @param depth_map 输出深度图 [width * height]
 * @param width 图像宽度
 * @param height 图像高度
 * @param calibration 立体标定参数（提供baseline和fx）
 * @return 0成功，-1失败
 */
int stereo_disparity_to_depth_map(const float* disparity_map,
                                   float* depth_map,
                                   int width, int height,
                                   const StereoCalibration* calibration);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_STEREO_CALIBRATION_H