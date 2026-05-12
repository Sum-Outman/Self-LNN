/**
 * @file knowledge_integration.h
 * @brief 知识库集成接口
 * 
 * 提供知识库、知识图谱、语义网络和推理引擎之间的集成功能。
 * 支持数据转换、统一查询和协同推理。
 */

#ifndef SELFLNN_KNOWLEDGE_INTEGRATION_H
#define SELFLNN_KNOWLEDGE_INTEGRATION_H

#include "selflnn/knowledge/knowledge.h"
#include "selflnn/knowledge/knowledge_graph.h"
#include "selflnn/knowledge/semantic_network.h"
#include "selflnn/knowledge/logic_reasoning.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 集成系统句柄
 */
typedef struct KnowledgeIntegrationSystem KnowledgeIntegrationSystem;

/**
 * @brief 集成配置
 */
typedef struct {
    int enable_auto_sync;            /**< 是否启用自动同步 */
    int sync_interval_ms;            /**< 同步间隔（毫秒） */
    float consistency_threshold;     /**< 一致性阈值 */
    int enable_cross_reasoning;      /**< 是否启用跨系统推理 */
} IntegrationConfig;

/**
 * @brief 创建知识集成系统
 * 
 * @param config 集成配置
 * @return KnowledgeIntegrationSystem* 集成系统句柄，失败返回NULL
 */
KnowledgeIntegrationSystem* knowledge_integration_create(const IntegrationConfig* config);

/**
 * @brief 释放知识集成系统
 * 
 * @param system 集成系统句柄
 */
void knowledge_integration_free(KnowledgeIntegrationSystem* system);

/**
 * @brief 注册知识库到集成系统
 * 
 * @param system 集成系统句柄
 * @param kb 知识库句柄
 * @param name 知识库名称
 * @return int 成功返回0，失败返回-1
 */
int knowledge_integration_register_kb(KnowledgeIntegrationSystem* system,
                                     KnowledgeBase* kb, const char* name);

/**
 * @brief 注册知识图谱到集成系统
 * 
 * @param system 集成系统句柄
 * @param graph 知识图谱句柄
 * @param name 知识图谱名称
 * @return int 成功返回0，失败返回-1
 */
int knowledge_integration_register_graph(KnowledgeIntegrationSystem* system,
                                        KnowledgeGraph* graph, const char* name);

/**
 * @brief 注册语义网络到集成系统
 * 
 * @param system 集成系统句柄
 * @param network 语义网络句柄
 * @param name 语义网络名称
 * @return int 成功返回0，失败返回-1
 */
int knowledge_integration_register_semantic_network(KnowledgeIntegrationSystem* system,
                                                   SemanticNetwork* network, const char* name);

/**
 * @brief 注册推理引擎到集成系统
 * 
 * @param system 集成系统句柄
 * @param engine 推理引擎句柄
 * @param name 推理引擎名称
 * @return int 成功返回0，失败返回-1
 */
int knowledge_integration_register_reasoning_engine(KnowledgeIntegrationSystem* system,
                                                   ReasoningEngine* engine, const char* name);

/**
 * @brief 同步所有注册系统
 * 
 * @param system 集成系统句柄
 * @return int 成功返回同步的条目数，失败返回-1
 */
int knowledge_integration_sync_all(KnowledgeIntegrationSystem* system);

/**
 * @brief 从知识库同步到知识图谱
 * 
 * @param system 集成系统句柄
 * @param kb_name 知识库名称
 * @param graph_name 知识图谱名称
 * @return int 成功返回同步的条目数，失败返回-1
 */
int knowledge_integration_sync_kb_to_graph(KnowledgeIntegrationSystem* system,
                                          const char* kb_name, const char* graph_name);

/**
 * @brief 从知识图谱同步到语义网络
 * 
 * @param system 集成系统句柄
 * @param graph_name 知识图谱名称
 * @param network_name 语义网络名称
 * @return int 成功返回同步的条目数，失败返回-1
 */
int knowledge_integration_sync_graph_to_network(KnowledgeIntegrationSystem* system,
                                               const char* graph_name, const char* network_name);

/**
 * @brief 从语义网络同步到知识库
 * 
 * @param system 集成系统句柄
 * @param network_name 语义网络名称
 * @param kb_name 知识库名称
 * @return int 成功返回同步的条目数，失败返回-1
 */
int knowledge_integration_sync_network_to_kb(KnowledgeIntegrationSystem* system,
                                            const char* network_name, const char* kb_name);

/**
 * @brief 统一查询
 * 
 * @param system 集成系统句柄
 * @param query 查询字符串
 * @param max_results 最大结果数
 * @param results 结果输出缓冲区
 * @return size_t 返回匹配的结果数
 */
size_t knowledge_integration_unified_query(KnowledgeIntegrationSystem* system,
                                          const char* query, size_t max_results,
                                          KnowledgeEntry* results);

/**
 * @brief 协同推理
 * 
 * @param system 集成系统句柄
 * @param premises 前提条件数组
 * @param premise_count 前提条件数量
 * @param max_inferences 最大推理数量
 * @param results 推理结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回推理出的结果数
 */
size_t knowledge_integration_cooperative_reasoning(KnowledgeIntegrationSystem* system,
                                                  const char** premises, size_t premise_count,
                                                  size_t max_inferences,
                                                  KnowledgeEntry* results, size_t max_results);

/**
 * @brief AGM信念状态
 */
typedef enum {
    BELIEF_STATUS_ACCEPTED = 0,
    BELIEF_STATUS_REJECTED = 1,
    BELIEF_STATUS_UNDETERMINED = 2
} BeliefStatus;

/**
 * @brief AGM信念状态记录
 */
typedef struct {
    int entry_id;
    float entrenchment_degree;
    BeliefStatus status;
    long timestamp;
} BeliefState;

/**
 * @brief AGM信念扩张——直接添加新信念，不检查一致性
 * 
 * @param system 集成系统句柄
 * @param kb 目标知识库
 * @param entry 要添加的知识条目
 * @return int 成功返回条目ID，失败返回-1
 */
int knowledge_integration_belief_expand(KnowledgeIntegrationSystem* system,
                                       KnowledgeBase* kb, const KnowledgeEntry* entry);

/**
 * @brief AGM信念收缩——移除信念及其逻辑依赖项
 * 
 * @param system 集成系统句柄
 * @param kb 目标知识库
 * @param entry_id 要收缩的条目ID
 * @return int 成功返回0，失败返回-1
 */
int knowledge_integration_belief_contract(KnowledgeIntegrationSystem* system,
                                         KnowledgeBase* kb, int entry_id);

/**
 * @brief AGM信念修正——使用Levi恒等式：先收缩矛盾信念，再扩张新信念
 * 
 * @param system 集成系统句柄
 * @param kb 目标知识库
 * @param entry 要添加的新知识条目
 * @param conflicting_entry 与之冲突的已有条目（可NULL，自动检测）
 * @return int 成功返回新条目ID，失败返回-1
 */
int knowledge_integration_belief_revise(KnowledgeIntegrationSystem* system,
                                       KnowledgeBase* kb, const KnowledgeEntry* entry,
                                       const KnowledgeEntry* conflicting_entry);

/**
 * @brief 检测知识库中的逻辑矛盾（直接矛盾+传递矛盾+规则事实冲突）
 * 
 * @param system 集成系统句柄
 * @param kb 目标知识库
 * @param conflict_pairs 矛盾对输出缓冲区（每对两个entry_id）
 * @param max_pairs 最大矛盾对数量
 * @return size_t 返回发现的矛盾对数量
 */
size_t knowledge_integration_detect_logical_contradictions(KnowledgeIntegrationSystem* system,
                                                          KnowledgeBase* kb,
                                                          int* conflict_pairs,
                                                          size_t max_pairs);

/**
 * @brief 检查系统间一致性
 * 
 * @param system 集成系统句柄
 * @param inconsistencies 不一致性输出缓冲区
 * @param max_inconsistencies 最大不一致性数量
 * @return size_t 返回发现的不一致性数量
 */
size_t knowledge_integration_check_consistency(KnowledgeIntegrationSystem* system,
                                              char** inconsistencies,
                                              size_t max_inconsistencies);

/**
 * @brief 获取集成系统统计信息
 * 
 * @param system 集成系统句柄
 * @param kb_count 知识库数量输出
 * @param graph_count 知识图谱数量输出
 * @param network_count 语义网络数量输出
 * @param engine_count 推理引擎数量输出
 * @return int 成功返回0，失败返回-1
 */
int knowledge_integration_get_stats(KnowledgeIntegrationSystem* system,
                                   size_t* kb_count, size_t* graph_count,
                                   size_t* network_count, size_t* engine_count);

/**
 * @brief 保存集成系统状态
 * 
 * @param system 集成系统句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int knowledge_integration_save_state(KnowledgeIntegrationSystem* system,
                                    const char* filename);

/**
 * @brief 加载集成系统状态
 * 
 * @param filename 文件名
 * @return KnowledgeIntegrationSystem* 集成系统句柄，失败返回NULL
 */
KnowledgeIntegrationSystem* knowledge_integration_load_state(const char* filename);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_KNOWLEDGE_INTEGRATION_H