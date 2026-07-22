/**
 * @file cfc_knowledge_embedding.h
 * @brief CfC知识图谱嵌入引擎接口（A03.1.3）
 *
 * 基于液态神经网络的全模态知识图谱嵌入：
 * 1. 连续空间嵌入 — TransE替代方案，用CfC ODE学习实体/关系嵌入
 * 2. 四元数旋转嵌入 — RotatE替代方案，用四元数旋转建模关系模式
 * 3. 液态图传播 — GNN替代方案，用CfC ODE在图上传播信息
 *
 * 100% 纯C实现，无第三方库依赖。
 */

#ifndef SELFLNN_CFC_KNOWLEDGE_EMBEDDING_H
#define SELFLNN_CFC_KNOWLEDGE_EMBEDDING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CFC_EMBED_MAX_ENTITIES     262144
#define CFC_EMBED_MAX_RELATIONS    65536
#define CFC_EMBED_MAX_DIM          1024
#define CFC_EMBED_DEFAULT_DIM      128

typedef enum {
    CFC_EMBED_CONTINUOUS = 0,
    CFC_EMBED_QUATERNION,
    CFC_EMBED_LIQUID_GRAPH,
    CFC_EMBED_HYBRID
} CfCEmbedType;

typedef struct {
    int num_entities;
    int num_relations;
    int max_entities;          /**< 实体最大容量，默认4096 */
    int max_relations;         /**< 关系最大容量，默认1024 */
    int embedding_dim;
    CfCEmbedType embed_type;
    float learning_rate;
    float margin;
    int num_negative_samples;
    int batch_size;
    int max_epochs;
    float cfc_tau;
    float cfc_dt;
    int cfc_steps;
    int enable_graph_propagation;
    int graph_prop_steps;
    float graph_prop_dropout;
    int use_quaternion;
} CfCEmbedConfig;

typedef struct CfCEmbedState CfCEmbedState;

CfCEmbedState* cfc_embed_create(const CfCEmbedConfig* config);
void cfc_embed_destroy(CfCEmbedState* state);

void cfc_embed_set_lnn_network(CfCEmbedState* state, void* lnn_network);
void* cfc_embed_get_lnn_network(const CfCEmbedState* state);

int cfc_embed_add_entity(CfCEmbedState* state, const char* entity_name);
int cfc_embed_add_relation(CfCEmbedState* state, const char* relation_name);
int cfc_embed_add_triple(CfCEmbedState* state, int head_id, int rel_id, int tail_id);

int cfc_embed_get_entity_id(const CfCEmbedState* state, const char* entity_name);
int cfc_embed_get_relation_id(const CfCEmbedState* state, const char* relation_name);

int cfc_embed_get_entity_embedding(const CfCEmbedState* state, int entity_id,
                                    float* embedding, int max_dim);
int cfc_embed_get_relation_embedding(const CfCEmbedState* state, int relation_id,
                                      float* embedding, int max_dim);

/**
 * @brief 训练嵌入模型
 *
 * @param state 引擎状态
 * @param epochs 训练轮数
 * @return int 成功返回0
 */
int cfc_embed_train(CfCEmbedState* state, int epochs);

/**
 * @brief 获取训练损失历史
 *
 * @param state 引擎状态
 * @param losses 输出损失数组
 * @param max_count 最大获取数量
 * @return int 返回获取的损失条目数
 */
int cfc_embed_get_loss_history(const CfCEmbedState* state, float* losses, int max_count);

/**
 * @brief 评分三元组
 *
 * 连续空间：score = ||h + r - t||₂
 * 四元数旋转：score = ||h ⊗ r - t||₂（四元数旋转）
 * 液态图：score = CFC_ODE(h, r, G)（图传播后的差异）
 *
 * @param state 引擎状态
 * @param head_id 头实体ID
 * @param rel_id 关系ID
 * @param tail_id 尾实体ID
 * @return float 评分（越小越合理）
 */
float cfc_embed_score_triple(CfCEmbedState* state, int head_id, int rel_id, int tail_id);

/**
 * @brief 知识图谱链接预测
 *
 * @param state 引擎状态
 * @param head_id 头实体ID
 * @param rel_id 关系ID
 * @param candidates 候选实体ID数组
 * @param num_candidates 候选数
 * @param scores 输出评分数组
 * @return int 成功返回0
 */
int cfc_embed_predict_tail(CfCEmbedState* state, int head_id, int rel_id,
                            const int* candidates, int num_candidates,
                            float* scores);

/**
 * @brief 液态图传播（多跳推理）
 *
 * 用CfC ODE在图上传播K步：
 *   h_v^{t+1} = h_v^t + CFC_ODE_sum(h_v^t, Σ_w h_w^t * r_vw)
 *
 * @param state 引擎状态
 * @param seed_entity 起始实体
 * @param steps 传播步数
 * @param result 输出传播后的嵌入
 * @param max_dim 最大维度
 * @return int 成功返回0
 */
int cfc_embed_graph_propagate(CfCEmbedState* state, int seed_entity,
                               int steps, float* result, int max_dim);

/**
 * @brief 保存嵌入模型
 *
 * @param state 引擎状态
 * @param filepath 文件路径
 * @return int 成功返回0
 */
int cfc_embed_save(const CfCEmbedState* state, const char* filepath);

/**
 * @brief 多跳推理：沿关系路径在嵌入空间中行走
 *
 * e0 + r1 + r2 + ... + rN → 预测实体嵌入
 *
 * @param state 嵌入状态
 * @param start_entity_id 起始实体ID
 * @param relation_path 关系路径ID数组
 * @param path_length 路径长度（跳数）
 * @param result_embedding 输出：最终实体嵌入 [dim]
 * @param intermediate_embeddings 可选：中间嵌入 [path_length+1][dim]
 * @param path_confidence 输出：路径置信度
 * @return int 成功返回0
 */
int cfc_embed_multi_hop_infer(const CfCEmbedState* state,
                               int start_entity_id,
                               const int* relation_path,
                               int path_length,
                               float* result_embedding,
                               float** intermediate_embeddings,
                               float* path_confidence);

/**
 * @brief 多跳推理查询：返回top-k匹配实体
 *
 * @param state 嵌入状态
 * @param start_entity_id 起始实体ID
 * @param relation_path 关系路径ID数组
 * @param path_length 路径长度
 * @param top_k 返回数量
 * @param matched_entity_ids 输出：匹配实体ID [top_k]
 * @param match_scores 输出：匹配分数 [top_k]
 * @return int 返回匹配数量
 */
int cfc_embed_multi_hop_query(const CfCEmbedState* state,
                               int start_entity_id,
                               const int* relation_path,
                               int path_length,
                               int top_k,
                               int* matched_entity_ids,
                               float* match_scores);

/**
 * @brief 加载嵌入模型
 *
 * @param filepath 文件路径
 * @return CfCEmbedState* 成功返回句柄，失败返回NULL
 */
CfCEmbedState* cfc_embed_load(const char* filepath);

/**
 * @brief 获取默认配置
 */
CfCEmbedConfig cfc_embed_default_config(void);

#ifdef __cplusplus
}
#endif

#endif
