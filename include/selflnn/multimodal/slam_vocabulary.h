/**
 * @file slam_vocabulary.h
 * @brief SLAM视觉词汇表与共视图模块接口
 *
 * 实现视觉词袋（BoW）模型：
 * - 词汇表树构建（K-means++聚类）
 * - TF-IDF权重计算
 * - BoW向量计算与相似度
 * - 共视图（Covisibility Graph）管理
 * - 本质图（Essential Graph）构建
 */
#ifndef SELFLNN_SLAM_VOCABULARY_H
#define SELFLNN_SLAM_VOCABULARY_H

#include "selflnn/multimodal/slam_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 词汇表树节点 ---- */

/**
 * @brief 创建词汇表树节点
 * @param descriptor_length 描述子维度
 * @return 新节点句柄，失败返回NULL
 */
VocabTreeNode* slam_vocab_node_create(int descriptor_length);

/**
 * @brief 递归释放词汇表树节点
 * @param node 节点句柄
 */
void slam_vocab_node_free(VocabTreeNode* node);

/**
 * @brief 训练词汇表树节点（递归K-means聚类）
 * @param node 节点句柄
 * @param descriptors 描述子数据 [num_descriptors × descriptor_length]
 * @param num_descriptors 描述子数量
 * @param descriptor_length 描述子维度
 * @param depth 当前深度
 * @param max_depth 最大深度
 * @param branch_factor 分支因子
 * @return 0=成功，-1=失败
 */
int slam_vocab_node_train(VocabTreeNode* node, const float* descriptors,
                          int num_descriptors, int descriptor_length,
                          int depth, int max_depth, int branch_factor);

/**
 * @brief 将描述子分配到最近叶子节点
 * @param node 根节点
 * @param descriptor 特征描述子
 * @param descriptor_length 描述子维度
 * @return 叶子节点索引，-1=失败
 */
int slam_vocab_node_assign(VocabTreeNode* node, const float* descriptor, int descriptor_length);

/* ---- 词汇表 ---- */

/**
 * @brief 初始化视觉词汇表
 * @param vocab 词汇表句柄
 * @param config 词汇表配置（可为NULL使用默认值）
 * @return 0=成功，-1=失败
 */
int slam_vocabulary_init(InternalVocabulary* vocab, const VisualVocabularyConfig* config);

/**
 * @brief 构建视觉词汇表
 * @param vocab 词汇表句柄
 * @param all_descriptors 所有训练描述子
 * @param num_descriptors 描述子数量
 * @param descriptor_length 描述子维度
 * @return 0=成功，-1=失败
 */
int slam_vocabulary_build(InternalVocabulary* vocab, const float* all_descriptors,
                          int num_descriptors, int descriptor_length);

/**
 * @brief 计算帧的BoW向量
 * @param vocab 词汇表句柄
 * @param descriptors 帧特征描述子
 * @param num_descriptors 描述子数量
 * @param descriptor_length 描述子维度
 * @param bow_vector 输出BoW向量
 * @param bow_vector_size 输入/输出BoW向量大小
 * @return 0=成功，-1=失败
 */
int slam_vocabulary_compute_bow(InternalVocabulary* vocab, const float* descriptors,
                               int num_descriptors, int descriptor_length,
                               float* bow_vector, int* bow_vector_size);

/**
 * @brief 计算两个BoW向量余弦相似度
 * @param bow1 BoW向量1
 * @param size1 向量1大小
 * @param bow2 BoW向量2
 * @param size2 向量2大小
 * @return 余弦相似度 [0-1]
 */
float slam_vocabulary_compute_similarity(const float* bow1, int size1,
                                        const float* bow2, int size2);

/**
 * @brief 添加帧描述子到词汇表统计
 * @param vocab 词汇表句柄
 * @param descriptors 帧特征描述子
 * @param num_descriptors 描述子数量
 * @param descriptor_length 描述子维度
 * @return 0=成功，-1=失败
 */
int slam_vocabulary_add_frame(InternalVocabulary* vocab, const float* descriptors,
                             int num_descriptors, int descriptor_length);

/**
 * @brief 更新TF-IDF权重
 * @param vocab 词汇表句柄
 * @return 0=成功，-1=失败
 */
int slam_vocabulary_update_tfidf(InternalVocabulary* vocab);

/**
 * @brief 释放词汇表资源
 * @param vocab 词汇表句柄
 */
void slam_vocabulary_free(InternalVocabulary* vocab);

/* ---- 共视图 ---- */

/**
 * @brief 初始化共视图
 * @param cov 共视图句柄
 * @param max_frames 最大帧数
 * @return 0=成功，-1=失败
 */
int slam_covisibility_init(InternalCovisibility* cov, int max_frames);

/**
 * @brief 释放共视图资源
 * @param cov 共视图句柄
 */
void slam_covisibility_free(InternalCovisibility* cov);

/**
 * @brief 更新共视图（基于共享路标点）
 * @param cov 共视图句柄
 * @param frame_id 帧ID
 * @param landmark_ids 路标ID数组
 * @param num_landmarks 路标数量
 * @param keyframes 关键帧数组
 * @param num_keyframes 关键帧数量
 * @return 0=成功，-1=失败
 */
int slam_covisibility_update(InternalCovisibility* cov, int frame_id,
                             int* landmark_ids, int num_landmarks,
                             const KeyFrame* keyframes, int num_keyframes);

/**
 * @brief 获取与指定帧共视的关键帧列表
 * @param cov 共视图句柄
 * @param frame_id 帧ID
 * @param connected_ids 输出连接帧ID数组
 * @param max_count 最大返回数量
 * @return 连接帧数量，-1=失败
 */
int slam_covisibility_get_connected(InternalCovisibility* cov, int frame_id,
                                    int* connected_ids, int max_count);

/**
 * @brief 获取两帧间共视权重
 * @param cov 共视图句柄
 * @param frame_id1 帧ID1
 * @param frame_id2 帧ID2
 * @return 共视权重（共享路标数），0=无连接
 */
int slam_covisibility_get_weight(InternalCovisibility* cov, int frame_id1, int frame_id2);

/**
 * @brief 从共视图构建本质图
 * @param cov 共视图句柄
 * @param keyframes 关键帧数组
 * @param num_keyframes 关键帧数量
 * @return 0=成功，-1=失败
 */
int slam_covisibility_build_essential_graph(InternalCovisibility* cov,
                                            const KeyFrame* keyframes,
                                            int num_keyframes);

/* ---- 聚类 ---- */

/**
 * @brief K-means++聚类初始化
 * @param descriptors 描述子数据
 * @param num_descriptors 描述子数量
 * @param descriptor_length 描述子维度
 * @param k 聚类数
 * @param centers 输出聚类中心 [k × descriptor_length]
 * @return 0=成功，-1=失败
 */
int slam_kmeans_plus_plus(const float* descriptors, int num_descriptors,
                          int descriptor_length, int k, float* centers);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SLAM_VOCABULARY_H */
