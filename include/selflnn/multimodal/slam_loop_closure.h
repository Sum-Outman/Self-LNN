/**
 * @file slam_loop_closure.h
 * @brief SLAM闭环检测与修正模块接口
 *
 * 实现完整的回环检测与修正流程：
 * - 基础矩阵/本质矩阵计算（8点法+RANSAC）
 * - Sampson距离几何验证
 * - 时间一致性检查
 * - BoW/混合候选选择
 * - 单应性矩阵验证
 * - 位姿图优化（Levenberg-Marquardt）
 * - 闭环地图融合与漂移传播
 */
#ifndef SELFLNN_SLAM_LOOP_CLOSURE_H
#define SELFLNN_SLAM_LOOP_CLOSURE_H

#include "selflnn/multimodal/slam_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 几何验证 ---- */

/**
 * @brief 8点法计算基础矩阵
 * @param points1 图像1点集 [x1,y1, x2,y2, ...]
 * @param points2 图像2点集
 * @param num_points 点对数（≥8）
 * @param F 输出3×3基础矩阵
 * @return 0=成功，-1=失败
 */
int slam_compute_fundamental_matrix_8point(const float* points1, const float* points2,
                                            int num_points, float* F);

/**
 * @brief 计算Sampson距离（点到对极线的几何误差）
 * @param F 3×3基础矩阵
 * @param x1 点1 x坐标
 * @param y1 点1 y坐标
 * @param x2 点2 x坐标
 * @param y2 点2 y坐标
 * @return Sampson距离
 */
float slam_compute_sampson_distance(const float* F, float x1, float y1, float x2, float y2);

/**
 * @brief 使用8点法+RANSAC进行回环几何验证
 * @param system SLAM系统句柄
 * @param frame_id 当前帧ID
 * @param candidate_id 候选帧ID
 * @param num_inliers 输出内点数量
 * @param inlier_ratio 输出内点比例
 * @param fundamental_matrix 输出基础矩阵
 * @return 1=通过验证，0=未通过，-1=失败
 */
int slam_verify_loop_geometric_8point(SlamSystem* system, int frame_id, int candidate_id,
                                      int* num_inliers, float* inlier_ratio,
                                      float* fundamental_matrix);

/**
 * @brief 单应性矩阵内点检测
 * @param f1 图像1特征点
 * @param f2 图像2特征点
 * @param matches 匹配对
 * @param num_matches 匹配数量
 * @param H 输入/输出3×3单应性矩阵
 * @param threshold 内点距离阈值
 * @param inlier_mask 输出内点掩码
 * @param max_inliers 最大内点数
 * @return 内点数量，-1=失败
 */
int slam_find_homography_inliers(const FeaturePoint* f1, const FeaturePoint* f2,
                                 const FeatureMatch* matches, int num_matches,
                                 float* H, float threshold, int* inlier_mask, int max_inliers);

/* ---- 时间一致性 ---- */

/**
 * @brief 时间一致性检查
 * @param system SLAM系统句柄
 * @param candidate_frame_id 候选帧ID
 * @return 1=一致，0=不一致
 */
int slam_temporal_consistency_check(SlamSystem* system, int candidate_frame_id);

/* ---- 闭环检测与修正 ---- */

/**
 * @brief 闭环检测主函数（BoW+几何验证+时间一致性）
 * @param system SLAM系统句柄
 * @param frame_id 当前帧ID
 * @param matched_frame_id 输出匹配帧ID
 * @return 1=检测到闭环，0=未检测到，-1=失败
 */
int slam_detect_loop_closure(SlamSystem* system, int frame_id, int* matched_frame_id);

/**
 * @brief 闭环修正（位姿图优化+地图融合+漂移传播）
 * @param system SLAM系统句柄
 * @param frame_id 当前帧ID
 * @param matched_frame_id 匹配帧ID
 * @return 1=修正成功，0=无需修正，-1=失败
 */
int slam_correct_loop_closure(SlamSystem* system, int frame_id, int matched_frame_id);

/**
 * @brief 闭环地图融合（合并重复路标点）
 * @param system SLAM系统句柄
 * @param frame_id 当前帧ID
 * @param matched_frame_id 匹配帧ID
 * @return 融合的路标点数量，-1=失败
 */
int slam_fuse_loop_closure_map(SlamSystem* system, int frame_id, int matched_frame_id);

/**
 * @brief 漂移修正传播到所有关键帧和路标点
 * @param system SLAM系统句柄
 * @param matched_frame_id 匹配帧ID
 * @param current_frame_id 当前帧ID
 * @param corrected_poses 已修正的位姿
 * @param num_corrected 已修正数量
 * @return 1=成功，-1=失败
 */
int slam_propagate_drift_correction(SlamSystem* system, int matched_frame_id,
                                    int current_frame_id, const float* corrected_poses,
                                    int num_corrected);

/* ---- BoW候选选择 ---- */

/**
 * @brief 计算帧的BoW向量
 * @param system SLAM系统句柄
 * @param frame_id 帧ID
 * @return 0=成功，-1=失败
 */
int slam_compute_bow_vector(SlamSystem* system, int frame_id);

/**
 * @brief 基于BoW相似度选择闭环候选
 * @param system SLAM系统句柄
 * @param frame_id 当前帧ID
 * @param candidates 输出候选帧ID数组
 * @param max_candidates 最大候选数
 * @param scores 输出候选分数数组
 * @return 候选数量
 */
int slam_select_candidates_by_bow(SlamSystem* system, int frame_id,
                                  int* candidates, int max_candidates, float* scores);

/**
 * @brief 混合候选选择（BoW+几何验证）
 * @param system SLAM系统句柄
 * @param frame_id 当前帧ID
 * @param candidates 输出候选帧ID数组
 * @param max_candidates 最大候选数
 * @param scores 输出候选分数数组
 * @return 候选数量
 */
int slam_select_candidates_hybrid(SlamSystem* system, int frame_id,
                                  int* candidates, int max_candidates, float* scores);

/* ---- 位姿图优化 ---- */

/**
 * @brief Levenberg-Marquardt位姿图优化
 * @param nodes 位姿图节点数组
 * @param num_nodes 节点数量
 * @param constraints 约束数组
 * @param num_constraints 约束数量
 * @param max_iterations 最大迭代次数（建议10-50）
 * @param lambda_init 初始阻尼因子（建议0.01）
 * @return 总残差平方和，负值=失败
 */
float slam_pose_graph_optimize(PoseGraphNode* nodes, int num_nodes,
                                const PoseGraphConstraint* constraints, int num_constraints,
                                int max_iterations, float lambda_init);

/**
 * @brief 从SLAM系统关键帧构建位姿图并优化
 * @param system SLAM系统句柄
 * @return 收敛后的总残差，负值=失败
 */
float slam_optimize_pose_graph_from_keyframes(SlamSystem* system);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SLAM_LOOP_CLOSURE_H */
