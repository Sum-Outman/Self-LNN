/**
 * @file depth_estimation.h
 * @brief 深度估计算法接口
 * 
 * 深度估计模块，支持单目深度估计和立体视觉深度估计。
 * 实现完整的深度感知能力，包括双目相机标定、立体校正、视差图计算。
 */

#ifndef SELFLNN_DEPTH_ESTIMATION_H
#define SELFLNN_DEPTH_ESTIMATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 深度估计方法枚举
 */
typedef enum {
    DEPTH_METHOD_MONOCULAR = 0,      /**< 单目深度估计（基于学习或几何） */
    DEPTH_METHOD_STEREO = 1,         /**< 立体视觉深度估计（双目匹配） */
    DEPTH_METHOD_HYBRID = 2,         /**< 混合方法 */
} DepthEstimationMethod;

/**
 * @brief 立体匹配算法枚举
 */
typedef enum {
    STEREO_MATCHING_BM = 0,          /**< 块匹配算法（Block Matching） */
    STEREO_MATCHING_SGBM = 1,        /**< 半全局块匹配算法（Semi-Global Block Matching） */
    STEREO_MATCHING_NCC = 2,         /**< 归一化互相关算法（Normalized Cross-Correlation） */
} StereoMatchingAlgorithm;

/**
 * @brief 相机标定参数结构体
 */
typedef struct {
    float fx, fy;           /**< 焦距（像素单位） */
    float cx, cy;           /**< 主点坐标（像素单位） */
    float k1, k2, k3;       /**< 径向畸变系数 */
    float p1, p2;           /**< 切向畸变系数 */
    float baseline;         /**< 基线长度（米），仅立体相机 */
    int image_width;        /**< 图像宽度（像素） */
    int image_height;       /**< 图像高度（像素） */
} CameraCalibration;

/* StereoCalibration 定义在 stereo_calibration.h 中 */
#include "stereo_calibration.h"

/**
 * @brief 深度估计配置结构体
 */
typedef struct {
    DepthEstimationMethod method;            /**< 深度估计方法 */
    StereoMatchingAlgorithm stereo_algorithm; /**< 立体匹配算法（如果使用立体视觉） */
    int enable_filtering;                    /**< 是否启用深度图滤波 */
    int enable_postprocessing;               /**< 是否启用后处理 */
    int enable_stereo_depth;                 /**< 是否启用立体深度估计（如果可用） */
    int disparity_range;                     /**< 视差搜索范围（像素） */
    int window_size;                         /**< 匹配窗口大小（像素） */
    int max_features;                        /**< 最大特征点数（用于特征匹配） */
    float min_depth;                         /**< 最小深度（米） */
    float max_depth;                         /**< 最大深度（米） */
    int use_gpu;                             /**< 是否使用GPU加速（如果可用） */
    int output_format;                       /**< 输出格式：0=深度图，1=点云，2=两者 */
} DepthEstimationConfig;

/**
 * @brief 深度估计结果结构体
 */
typedef struct {
    float* depth_map;               /**< 深度图数据（行主序，单位：米） */
    float* disparity_map;           /**< 视差图数据（行主序，单位：像素） */
    float* point_cloud;             /**< 点云数据（XYZ坐标，每点3个浮点数） */
    int width;                      /**< 深度图宽度 */
    int height;                     /**< 深度图高度 */
    size_t point_count;             /**< 点云点数 */
    float depth_accuracy;           /**< 深度估计准确度（均方根误差，如果已知） */
    float processing_time_ms;       /**< 处理时间（毫秒） */
} DepthEstimationResult;

/**
 * @brief CfC ODE深度估计网络配置结构体
 */
typedef struct {
    int input_width;                /**< 输入图像宽度 */
    int input_height;               /**< 输入图像高度 */
    int input_channels;             /**< 输入图像通道数 */
    int patch_size;                 /**< 分块大小 */
    int hidden_dim;                 /**< CfC隐藏状态维度 */
    int num_layers;                 /**< CfC层数 */
    float dt;                       /**< ODE时间步长 */
    float tau_min;                  /**< 最小时间常数 */
    float tau_max;                  /**< 最大时间常数 */
    int use_adaptive_tau;           /**< 是否使用自适应时间常数 */
    int num_ode_steps;              /**< ODE积分步数 */
    char* model_weights;            /**< 预训练权重路径（如果可用） */
} CfcDepthConfig;

/**
 * @brief 深度估计处理器句柄
 */
typedef struct DepthEstimator DepthEstimator;

/**
 * @brief 创建深度估计处理器
 * 
 * @param config 深度估计配置
 * @return DepthEstimator* 处理器句柄，失败返回NULL
 */
DepthEstimator* depth_estimator_create(const DepthEstimationConfig* config);

/**
 * @brief 释放深度估计处理器
 * 
 * @param estimator 处理器句柄
 */
void depth_estimator_free(DepthEstimator* estimator);

/**
 * @brief 执行单目深度估计
 * 
 * @param estimator 处理器句柄
 * @param image 输入图像数据（行主序，RGB或灰度）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数（1=灰度，3=RGB）
 * @param calibration 相机标定参数（可为NULL，如果未知）
 * @param result 深度估计结果输出（需要预先分配）
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_monocular(DepthEstimator* estimator,
                            const float* image, int width, int height, int channels,
                            const CameraCalibration* calibration,
                            DepthEstimationResult* result);

/**
 * @brief 执行立体视觉深度估计
 * 
 * @param estimator 处理器句柄
 * @param left_image 左图像数据
 * @param right_image 右图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param calibration 立体相机标定参数
 * @param result 深度估计结果输出
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_stereo(DepthEstimator* estimator,
                         const float* left_image, const float* right_image,
                         int width, int height, int channels,
                         const StereoCalibration* calibration,
                         DepthEstimationResult* result);

/**
 * @brief 标定双目相机
 * 
 * @param estimator 处理器句柄
 * @param left_images 左相机图像数组
 * @param right_images 右相机图像数组
 * @param num_images 图像数量
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param pattern_width 标定板角点宽度（内角点数）
 * @param pattern_height 标定板角点高度（内角点数）
 * @param square_size 标定板方格大小（米）
 * @param calibration 标定结果输出
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_calibrate_stereo(DepthEstimator* estimator,
                                   const float** left_images, const float** right_images,
                                   int num_images, int width, int height, int channels,
                                   int pattern_width, int pattern_height, float square_size,
                                   StereoCalibration* calibration);

/**
 * @brief 保存相机标定参数到文件
 * 
 * @param calibration 标定参数
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_save_calibration(const StereoCalibration* calibration, const char* filepath);

/**
 * @brief 从文件加载相机标定参数
 * 
 * @param calibration 标定参数输出
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_load_calibration(StereoCalibration* calibration, const char* filepath);

/**
 * @brief 初始化单目深度估计网络
 * 
 * @param estimator 处理器句柄
 * @param config 网络配置
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_init_monocular_network(DepthEstimator* estimator,
                                         const CfcDepthConfig* config);

/**
 * @brief 训练单目深度估计网络
 * 
 * @param estimator 处理器句柄
 * @param training_images 训练图像数组
 * @param ground_truth_depths 真实深度图数组
 * @param num_samples 样本数量
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param num_epochs 训练轮数
 * @param learning_rate 学习率
 * @param batch_size 批次大小
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_train_monocular_network(DepthEstimator* estimator,
                                          const float** training_images,
                                          const float** ground_truth_depths,
                                          int num_samples, int width, int height, int channels,
                                          int num_epochs, float learning_rate, int batch_size);

/**
 * @brief 保存深度估计模型
 * 
 * @param estimator 处理器句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_save_model(const DepthEstimator* estimator, const char* filepath);

/**
 * @brief 加载深度估计模型
 * 
 * @param estimator 处理器句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_load_model(DepthEstimator* estimator, const char* filepath);

/**
 * @brief 获取深度估计配置
 * 
 * @param estimator 处理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int depth_estimator_get_config(const DepthEstimator* estimator, DepthEstimationConfig* config);

/**
 * @brief 设置深度估计配置
 * 
 * @param estimator 处理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int depth_estimator_set_config(DepthEstimator* estimator, const DepthEstimationConfig* config);

/**
 * @brief 重置深度估计处理器
 * 
 * @param estimator 处理器句柄
 */
void depth_estimator_reset(DepthEstimator* estimator);

/**
 * @brief 将深度图转换为点云
 * 
 * @param depth_map 深度图数据
 * @param width 深度图宽度
 * @param height 深度图高度
 * @param calibration 相机标定参数
 * @param point_cloud 点云输出缓冲区
 * @param max_points 最大点数
 * @return int 成功返回点数，失败返回-1
 */
int depth_estimate_convert_to_point_cloud(const float* depth_map, int width, int height,
                                         const CameraCalibration* calibration,
                                         float* point_cloud, size_t max_points);

/**
 * @brief 滤波深度图
 * 
 * @param depth_map 深度图数据（输入输出）
 * @param width 宽度
 * @param height 高度
 * @param kernel_size 滤波核大小
 * @param sigma 高斯滤波标准差
 * @param bilateral 是否使用双边滤波
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_filter_depth_map(float* depth_map, int width, int height,
                                   int kernel_size, float sigma, int bilateral);

/**
 * @brief 计算深度估计误差
 * 
 * @param estimated_depth 估计深度图
 * @param ground_truth_depth 真实深度图
 * @param width 宽度
 * @param height 高度
 * @param valid_mask 有效像素掩码（可选）
 * @param rmse 均方根误差输出
 * @param mae 平均绝对误差输出
 * @param accuracy 准确度输出（百分比）
 * @return int 成功返回0，失败返回-1
 */
int depth_estimate_compute_error(const float* estimated_depth, const float* ground_truth_depth,
                                int width, int height, const unsigned char* valid_mask,
                                float* rmse, float* mae, float* accuracy);

/**
 * @brief 获取默认的深度估计配置
 * 
 * @return DepthEstimationConfig 默认配置
 */
DepthEstimationConfig depth_estimation_get_default_config(void);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_DEPTH_ESTIMATION_H