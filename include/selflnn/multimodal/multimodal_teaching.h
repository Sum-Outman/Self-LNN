#ifndef SELFLNN_MULTIMODAL_TEACHING_H
#define SELFLNN_MULTIMODAL_TEACHING_H

/**
 * @file multimodal_teaching.h
 * @brief 多模态教学系统接口
 *
 * 通过多模态演示进行模仿学习和技能教学。
 * 所有模态特征通过简单拼接后送入同一个CfC连续动态系统进行状态演化，
 * 不产生跨模态注意力交互。
 */

#include "selflnn/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TEACH_VISUAL_FEAT_DIM 384       /**< 视觉特征维度 */
#define TEACH_AUDIO_FEAT_DIM 256        /**< 音频特征维度 */
#define TEACH_TEXT_FEAT_DIM 256         /**< 文本特征维度 */
#define TEACH_SENSOR_FEAT_DIM 128       /**< 传感器特征维度 */
#define TEACH_FUSED_FEAT_DIM 1024       /**< 融合特征维度 (384+256+256+128=1024 精确对齐) */
#define TEACH_MODALITIES 4              /**< 模态数量 */
#define TEACH_MAX_PRIMITIVES 64         /**< 最大基础动作原语数 */
#define TEACH_PRIMITIVE_FEAT_DIM 128    /**< 原语特征维度 */
#define TEACH_MAX_DEMONSTRATIONS 32     /**< 最大演示数 */
#define TEACH_MAX_SKILLS 64             /**< 最大技能数 */
#define TEACH_EVAL_METRICS 6            /**< 评估指标数 */
#define TEACH_ALIGNMENT_ITERATIONS 100  /**< 对齐迭代次数 */
#define TEACH_REPLAY_BUFFER_SIZE 1024   /**< 回放缓冲区大小 */

/**
 * @brief 模态类型枚举
 */
typedef enum {
    TEACH_MODAL_VISION = 0,             /**< 视觉模态 */
    TEACH_MODAL_AUDIO = 1,              /**< 音频模态 */
    TEACH_MODAL_TEXT = 2,               /**< 文本模态 */
    TEACH_MODAL_SENSOR = 3              /**< 传感器模态 */
} TeachModalType;

/**
 * @brief 多模态教学帧
 *
 * 包含一帧中所有模态的特征数据和融合特征。
 * 所有特征通过统一信号处理器处理后拼接为融合特征，再送入CfC系统。
 */
typedef struct {
    float visual_feat[TEACH_VISUAL_FEAT_DIM];   /**< 视觉特征向量 */
    float audio_feat[TEACH_AUDIO_FEAT_DIM];     /**< 音频特征向量 */
    float text_feat[TEACH_TEXT_FEAT_DIM];       /**< 文本特征向量 */
    float sensor_feat[TEACH_SENSOR_FEAT_DIM];   /**< 传感器特征向量 */
    float fused_feat[TEACH_FUSED_FEAT_DIM];     /**< 拼接融合特征 */
    float timestamp;                            /**< 时间戳 */
    int has_visual;                             /**< 是否有视觉数据 */
    int has_audio;                              /**< 是否有音频数据 */
    int has_text;                               /**< 是否有文本数据 */
    int has_sensor;                             /**< 是否有传感器数据 */
} TeachModalFrame;

/**
 * @brief 多模态序列
 *
 * 一组按时间排列的教学帧，构成一个完整的演示序列。
 */
typedef struct {
    TeachModalFrame* frames;            /**< 帧数组 */
    size_t num_frames;                  /**< 当前帧数 */
    size_t max_frames;                  /**< 最大帧数 */
    size_t visual_width;                /**< 视觉宽度 */
    size_t visual_height;               /**< 视觉高度 */
    size_t audio_sample_rate;           /**< 音频采样率 */
    int is_complete;                    /**< 是否已完成录制 */
} TeachModalSequence;

/**
 * @brief 教学融合配置
 *
 * 控制各模态权重和时序融合参数。
 * 权重仅用于最终输出加权，不影响CfC内部状态演化。
 */
typedef struct {
    float visual_weight;                /**< 视觉权重（0.0-1.0） */
    float audio_weight;                 /**< 音频权重（0.0-1.0） */
    float text_weight;                  /**< 文本权重（0.0-1.0） */
    float sensor_weight;                /**< 传感器权重（0.0-1.0） */
    float temporal_decay;               /**< 时序衰减系数 */
    int use_temporal_fusion;            /**< 是否使用时序融合 */
    float fusion_temperature;           /**< 融合温度参数 */
} TeachFusionConfig;

/** @brief 默认教学融合配置 */
#define TEACH_FUSION_CONFIG_DEFAULT { \
    0.4f, 0.2f, 0.25f, 0.15f, 0.95f, 1, 0.5f \
}

/** @brief 多模态教学系统句柄（不透明类型） */
typedef struct MultimodalTeachingSystem MultimodalTeachingSystem;

/**
 * @brief 创建多模态教学系统
 * @param config 融合配置
 * @return 教学系统句柄，失败返回NULL
 */
MultimodalTeachingSystem* multimodal_teaching_create(TeachFusionConfig config);

/**
 * @brief 销毁多模态教学系统
 * @param system 教学系统句柄
 */
void multimodal_teaching_destroy(MultimodalTeachingSystem* system);

/**
 * @brief 注入一帧多模态数据到教学序列
 * @param system 教学系统句柄
 * @param visual_data 视觉特征数据
 * @param visual_size 视觉特征维度
 * @param audio_data 音频特征数据
 * @param audio_size 音频特征维度
 * @param text_data 文本特征数据
 * @param text_size 文本特征维度
 * @param sensor_data 传感器特征数据
 * @param sensor_size 传感器特征维度
 * @param timestamp 时间戳
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_ingest_frame(MultimodalTeachingSystem* system,
                                       const float* visual_data,
                                       size_t visual_size,
                                       const float* audio_data,
                                       size_t audio_size,
                                       const float* text_data,
                                       size_t text_size,
                                       const float* sensor_data,
                                       size_t sensor_size,
                                       float timestamp);

/**
 * @brief 融合指定序列的帧数据
 * @param system 教学系统句柄
 * @param sequence_id 序列ID
 * @param fused_output [输出] 融合特征输出
 * @param fused_dim 融合特征维度
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_fuse_sequence(MultimodalTeachingSystem* system,
                                        size_t sequence_id,
                                        float* fused_output,
                                        size_t fused_dim);

/**
 * @brief 将教学观察编码为教学嵌入向量
 * @param system 教学系统句柄
 * @param observations 观察序列
 * @param num_steps 时间步数
 * @param obs_dim 观察维度
 * @param teaching_embedding [输出] 教学嵌入向量
 * @param embed_dim 嵌入维度
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_encode_teaching(MultimodalTeachingSystem* system,
                                          const float* observations,
                                          size_t num_steps,
                                          size_t obs_dim,
                                          float* teaching_embedding,
                                          size_t embed_dim);

/**
 * @brief 跨模态检索（基于拼接特征的相似度匹配，非交叉注意力）
 * @param system 教学系统句柄
 * @param query_embedding 查询嵌入向量
 * @param query_dim 查询维度
 * @param query_modal 查询模态类型
 * @param retrieved_feats [输出] 检索到的特征
 * @param retrieved_dim 检索特征维度
 * @param num_retrieved [输出] 检索到的数量
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_cross_modal_retrieval(
    MultimodalTeachingSystem* system,
    const float* query_embedding,
    size_t query_dim,
    TeachModalType query_modal,
    float* retrieved_feats,
    size_t retrieved_dim,
    size_t* num_retrieved);

/**
 * @brief 获取指定序列
 * @param system 教学系统句柄
 * @param sequence_id 序列ID
 * @param sequence_out [输出] 序列数据
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_get_sequence(MultimodalTeachingSystem* system,
                                       size_t sequence_id,
                                       TeachModalSequence* sequence_out);

/**
 * @brief 清空所有教学数据
 * @param system 教学系统句柄
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_clear(MultimodalTeachingSystem* system);

/**
 * @brief 设置关联的LNN（液态神经网络）实例
 * @param system 教学系统句柄
 * @param lnn_net LNN实例指针
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_set_lnn(MultimodalTeachingSystem* system, void* lnn_net);

/**
 * @brief 从指定序列中学习运动模式
 * @param system 教学系统句柄
 * @param sequence_id 序列ID
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_learn_pattern(MultimodalTeachingSystem* system, size_t sequence_id);

/**
 * @brief 开始回放指定序列
 * @param system 教学系统句柄
 * @param sequence_id 序列ID
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_replay_start(MultimodalTeachingSystem* system, size_t sequence_id);

/**
 * @brief 执行回放一步
 * @param system 教学系统句柄
 * @param frame_out [输出] 当前帧数据
 * @param frame_dim 帧数据维度
 * @param done [输出] 是否回放完成
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_replay_step(MultimodalTeachingSystem* system,
                                      float* frame_out, size_t frame_dim, int* done);

/**
 * @brief 停止回放
 * @param system 教学系统句柄
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_replay_stop(MultimodalTeachingSystem* system);

/**
 * @brief 从序列中提取基础动作原语
 * @param system 教学系统句柄
 * @param sequence_id 序列ID
 * @param num_primitives [输出] 提取的原语数量
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_extract_primitives(MultimodalTeachingSystem* system,
                                            size_t sequence_id, size_t* num_primitives);

/**
 * @brief 获取指定原语的特征
 * @param system 教学系统句柄
 * @param primitive_id 原语ID
 * @param feat_out [输出] 原语特征
 * @param feat_dim 特征维度
 * @param num_frames [输出] 原语包含的帧数
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_get_primitive(MultimodalTeachingSystem* system,
                                       size_t primitive_id, float* feat_out,
                                       size_t feat_dim, size_t* num_frames);

/**
 * @brief 聚合多个演示序列为统一表示
 * @param system 教学系统句柄
 * @param sequence_ids 序列ID数组
 * @param num_seqs 序列数量
 * @param aggregated [输出] 聚合结果
 * @param agg_dim 聚合维度
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_aggregate_demonstrations(MultimodalTeachingSystem* system,
                                                  const size_t* sequence_ids,
                                                  size_t num_seqs, float* aggregated,
                                                  size_t agg_dim);

/**
 * @brief 评估指定序列的技能水平
 * @param system 教学系统句柄
 * @param sequence_id 序列ID
 * @param metrics [输出] 评估指标数组
 * @param metrics_dim 指标维度
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_evaluate_skill(MultimodalTeachingSystem* system,
                                        size_t sequence_id, float* metrics,
                                        size_t metrics_dim);

/**
 * @brief 增量学习新观察（基于CfC状态更新，非回放式训练）
 * @param system 教学系统句柄
 * @param new_obs 新观察序列
 * @param num_steps 时间步数
 * @param obs_dim 观察维度
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_incremental_learn(MultimodalTeachingSystem* system,
                                           const float* new_obs, size_t num_steps,
                                           size_t obs_dim);

/**
 * @brief 对齐各模态特征空间（通过统一信号处理器进行，非跨模态注意力）
 * @param system 教学系统句柄
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_align_modalities(MultimodalTeachingSystem* system);

/**
 * @brief 获取教学系统统计信息
 * @param system 教学系统句柄
 * @param num_sequences [输出] 序列总数
 * @param total_frames [输出] 总帧数
 * @param num_primitives [输出] 原语总数
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_get_statistics(MultimodalTeachingSystem* system,
                                        size_t* num_sequences,
                                        size_t* total_frames,
                                        size_t* num_primitives);

/**
 * @brief 导出指定序列数据
 * @param system 教学系统句柄
 * @param sequence_id 序列ID
 * @param export_data [输出] 导出数据缓冲区
 * @param export_dim 导出维度
 * @param export_size [输出] 实际导出大小
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_export_sequence(MultimodalTeachingSystem* system,
                                         size_t sequence_id, float* export_data,
                                         size_t export_dim, size_t* export_size);

/**
 * @brief 设置融合权重（仅用于最终输出加权，不影响CfC内部状态演化）
 * @param system 教学系统句柄
 * @param visual_w 视觉权重
 * @param audio_w 音频权重
 * @param text_w 文本权重
 * @param sensor_w 传感器权重
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_set_fusion_weights(MultimodalTeachingSystem* system,
                                            float visual_w, float audio_w,
                                            float text_w, float sensor_w);

/**
 * @brief 相似度搜索（基于拼接特征的余弦相似度）
 * @param system 教学系统句柄
 * @param query 查询向量
 * @param query_dim 查询维度
 * @param result_ids [输出] 结果ID数组
 * @param result_scores [输出] 结果相似度分数
 * @param max_results 最大结果数
 * @param num_results [输出] 实际结果数
 * @return 0=成功，非0=失败
 */
int multimodal_teaching_similarity_search(MultimodalTeachingSystem* system,
                                           const float* query, size_t query_dim,
                                           size_t* result_ids, float* result_scores,
                                           size_t max_results, size_t* num_results);

#ifdef __cplusplus
}
#endif

#endif
