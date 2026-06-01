#ifndef SELFLNN_SWARM_ENHANCED_H
#define SELFLNN_SWARM_ENHANCED_H

/**
 * @file swarm_enhanced.h
 * @brief 增强群体智能引擎 — 【高级增强功能（与基础版互补，非替代）】
 *
 * P3-004 功能边界说明:
 *   ✅ 本文件: 自适应ACO蚁群（自适应信息素更新）、自适应ABC蜂群
 *   ✅ 本文件: 分布式共识算法（Raft协议）、CfC液态通信集群
 *   ✅ 本文件: 自愈集群（节点故障检测+自动恢复）、拉普拉斯增强集群通信
 *   ✅ 本文件: 与 swarm_intelligence.h 基础版协同，提供高级集群智能能力
 *   ❌ 非本文件: 基础群体算法(ACO/ABC/PSO/GWO/WOA)、基础拓扑管理、收敛控制
 *                → 请使用 swarm_intelligence.h（基础功能实现）
 *
 *   基础版 (swarm_intelligence.h) → 群体智能核心算法 + 基础拓扑
 *   增强版 (swarm_enhanced.h)     → 自适应群体 + 分布式共识 + CfC液态通信 + 自愈
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SWARM_EPSILON 1e-8f
#define SWARM_INF 1e20f
#define SWARM_ENH_MAX_NODES 256
#define SWARM_ENH_MAX_COMMAND_SIZE 4096

#define SWARM_ENH_ACO_ADAPTIVE           0
#define SWARM_ENH_ABC_ADAPTIVE           1
#define SWARM_ENH_DISTRIBUTED_CONSENSUS  2
#define SWARM_ENH_LIQUID_COMM            3
#define SWARM_ENH_SELF_HEALING           4

#define CONSENSUS_RAFT 0

#define ACO_PHEROMONE_ANT_CYCLE 0

#define NODE_STATE_OFFLINE  -1
#define NODE_STATE_FOLLOWER  0
#define NODE_STATE_CANDIDATE 1
#define NODE_STATE_LEADER    2

typedef enum {
    NODE_STATE_ENUM_FOLLOWER  = 0,
    NODE_STATE_ENUM_CANDIDATE = 1,
    NODE_STATE_ENUM_LEADER    = 2
} NodeState;

typedef enum {
    SWARM_ENHANCED_ACO = 0,
    SWARM_ENHANCED_ABC = 1,
    SWARM_ENHANCED_PSO = 2,
    SWARM_ENHANCED_CONSENSUS = 3,
    SWARM_ENHANCED_LIQUID_COMM = 4,
    SWARM_ENHANCED_SELF_HEALING = 5
} SwarmEnhancedAlgorithm;

typedef struct {
    int num_ants;
    int num_nodes;
    int max_iterations;
    float alpha;
    float beta;
    float evaporation_rate;
    float initial_pheromone;
    float pheromone_init;
    float q0;
    float elite_weight;
    int enable_cfc_path_selection;
    float cfc_time_constant;
    int cfc_steps;
    float cfc_tau;
    float cfc_dt;
    int update_strategy;
    int enable_adaptive_params;
    int enable_local_search;
} ACOEnhancedConfig;

typedef struct {
    int num_bees;
    int num_employed;
    int num_onlooker;
    int max_cycles;
    int max_trials;
    float mutation_rate;
    int enable_cfc_evaluation;
    float cfc_time_constant;
    int problem_dimension;
    float lower_bound;
    float upper_bound;
    int food_sources;
    float limit;
    float scout_probability;
    int cfc_steps;
    float cfc_tau;
    float cfc_dt;
    int colony_size;
    int enable_adaptive_search;
    int enable_multi_objective;
} ABCEnhancedConfig;

typedef struct {
    int num_nodes;
    int node_id;
    int num_sources;
    float* source_dimensions;
    float* target_weights;
    float convergence_threshold;
    int max_iterations;
    float cfc_time_constant;
    int enable_cfc_evaluation;
} SwarmConsensusConfig;

typedef struct {
    int node_id;
    int role;
    int port;
    float timeout_ms;
    int current_term;
    int election_timeout_min_ms;
    int election_timeout_max_ms;
} ConsensusNodeConfig;

typedef struct {
    int protocol_type;
    int current_term;
    int voted_for;
    int leader_id;
} ConsensusProtocol;

typedef struct {
    int index;
    int term;
    char* data;
    int data_size;
    int command_size;
    int command_type;
    int is_committed;
    char* command_data;
} ConsensusLogEntry;

typedef struct {
    int source_id;
    int target_id;
    int msg_type;
    float* state;
    float* state_vector;
    int state_dim;
    long timestamp;
} LiquidMessage;

typedef struct {
    int num_nodes;
    float heartbeat_interval;
    int max_missed;
    int enable_auto_healing;
} SelfHealingConfig;

typedef struct SwarmEnhancedEngine SwarmEnhancedEngine;

SwarmEnhancedEngine* swarm_aco_enhanced_create(const ACOEnhancedConfig* config);
void swarm_aco_enhanced_destroy(SwarmEnhancedEngine* engine);
ACOEnhancedConfig swarm_aco_default_config(void);
int swarm_aco_init_graph(SwarmEnhancedEngine* engine, int num_nodes, const float* distance_matrix);
int swarm_aco_iterate(SwarmEnhancedEngine* engine, int iteration);
int swarm_aco_get_best_path(SwarmEnhancedEngine* engine, int* path, int max_path_len, float* total_distance);
int swarm_aco_get_pheromone(SwarmEnhancedEngine* engine, float* pheromone_matrix, size_t max_size);

SwarmEnhancedEngine* swarm_abc_enhanced_create(const ABCEnhancedConfig* config);
int swarm_abc_init_sources(SwarmEnhancedEngine* engine, int dimensions, const float* lower_bound, const float* upper_bound);
int swarm_abc_iterate(SwarmEnhancedEngine* engine, int cycle);
int swarm_abc_get_best_solution(SwarmEnhancedEngine* engine, float* solution, int max_dim, float* fitness);

SwarmEnhancedEngine* swarm_consensus_create_node(const ConsensusNodeConfig* config, int protocol_type);
int swarm_consensus_register_peers(SwarmEnhancedEngine* engine, const int* node_ids, int num_nodes);
int swarm_consensus_start_election(SwarmEnhancedEngine* engine);
int swarm_consensus_append_entry(SwarmEnhancedEngine* engine, const ConsensusLogEntry* entry);
int swarm_consensus_commit_entry(SwarmEnhancedEngine* engine, long index);
int swarm_consensus_get_leader(SwarmEnhancedEngine* engine);
NodeState swarm_consensus_get_state(SwarmEnhancedEngine* engine);

/* H-015修复: Raft协议完整补充 — 心跳/日志复制/提交推进/选举超时/投票接收 */
int swarm_consensus_send_heartbeat(SwarmEnhancedEngine* engine);
int swarm_consensus_replicate_log(SwarmEnhancedEngine* engine);
int swarm_consensus_advance_commit_index(SwarmEnhancedEngine* engine);
int swarm_consensus_handle_request_vote(SwarmEnhancedEngine* engine, int candidate_id,
                                         float candidate_term, long candidate_log_index,
                                         long candidate_log_term);
int swarm_consensus_handle_append_entries(SwarmEnhancedEngine* engine, int leader_id,
                                           float leader_term, long prev_log_index,
                                           long prev_log_term, const ConsensusLogEntry* entries,
                                           int entry_count, long leader_commit);
int swarm_consensus_election_timeout_check(SwarmEnhancedEngine* engine);

int swarm_liquid_comm_create(SwarmEnhancedEngine* engine, int num_nodes, int state_dim, int hidden_dim);
int swarm_liquid_comm_send(SwarmEnhancedEngine* engine, int channel_id, const LiquidMessage* message);
int swarm_liquid_comm_receive(SwarmEnhancedEngine* engine, int channel_id, LiquidMessage* message, int block);
int swarm_liquid_comm_sync(SwarmEnhancedEngine* engine, int channel_id, int sync_steps);

int swarm_self_healing_configure(SwarmEnhancedEngine* engine, const SelfHealingConfig* config);
int swarm_self_healing_detect_failures(SwarmEnhancedEngine* engine, int* failed_nodes, int max_failures);
int swarm_self_healing_replace_node(SwarmEnhancedEngine* engine, int failed_node_id, int replacement_id);
int swarm_self_healing_redistribute_tasks(SwarmEnhancedEngine* engine, int failed_node_id);

int swarm_enhanced_save(const SwarmEnhancedEngine* engine, SwarmEnhancedAlgorithm algorithm, const char* filepath);
SwarmEnhancedEngine* swarm_enhanced_load(SwarmEnhancedAlgorithm algorithm, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif
