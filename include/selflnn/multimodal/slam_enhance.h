/**
 * @file slam_enhance.h
 * @brief SLAM增强模块接口 — 高级优化与多传感器融合
 *
 * 提供SLAM高级优化功能：
 * - slam_enhance_bundle_adjust: 束调整(Bundle Adjustment), Schur补+LM
 * - slam_enhance_imu_preintegrate: IMU预积分因子
 * - slam_enhance_pose_graph_optimize: 全局位姿图优化(PGO)
 *
 * 100%纯C接口，与slam.h主模块协同工作。
 */

#ifndef SELFLNN_SLAM_ENHANCE_H
#define SELFLNN_SLAM_ENHANCE_H

#include "selflnn/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 束调整优化
 *
 * 最小化重投影误差，联合优化相机参数和3D点位置。
 * 使用Levenberg-Marquardt算法 + Schur补消元。
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
 * 支持加速度计和陀螺仪零偏校正。
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
 * @brief 全局位姿图优化
 *
 * 使用非线性最小二乘优化回环约束下的全局位姿图。
 * 支持固定节点（锚点）约束。
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

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SLAM_ENHANCE_H */
