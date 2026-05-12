/**
 * @file swarm_enhanced.h
 * @brief 群体智能深度增强系统接口（A06.4）
 *
 * 在现有群体智能系统基础上，扩展深度增强功能：
 * 1. ACO增强版 — 自适应信息素更新 + CfC液态路径选择
 * 2. ABC增强版 — 自适应雇佣蜂 + CfC液态蜜源评估
 * 3. 分布式一致性算法 — 拜占庭容错共识 + RAFT简化版
 * 4. 群体液态通信 — CfC ODE群体状态同步
 * 5. 群体自愈与重组 — 故障节点自动替换
 * 6. CfC增强群体优化 — 液态粒子群演化
 *
 * 100% 纯C实现，无第三方库依赖。
 */

#ifndef SELFLNN_SWARM_ENHANCED_H
#define SELFLNN_SWARM_ENHANCED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 配置常量
 * ============================================================================ */
#define SWARM_ENH_MAX_NODES         4096    /**< 最大节点数 */
#define SWARM_ENH_MAX_EDGES         262144  /**< 最大边数 */
#define SWARM_ENH_MAX_PHEROMONE_TYPES 16   /**< 最大信息素类型数 */
#define SWARM_ENH_MAX_CONSENSUS_ROUNDS 1000 /**< 最大共识轮数 */
#define SWARM_ENH_MAX_COMMAND_SIZE  65536   /**< 最大命令大小 */

/* ============================================================================
 * 类型定义
 * ============================================================================ */

/**
 * @brief 增强群体算法类型
 */
typedef enum {
    SWARM_ENH_ACO_ADAPTIVE = 0,             /**< 自适应蚁群优化 */
    SWARM_ENH_ABC_ADAPTIVE,                  /**< 自适应蜂群优化 */
    SWARM_ENH_DISTRIBUTED_CONSENSUS,        /**< 分布式一致性 */
    SWARM_ENH_LIQUID_COMMUNICATION,         /**< 群体液态通信 */
    SWARM_ENH_SELF_HEALING,                 /**< 群体自愈 */
    SWARM_ENH_CFC_SWARM,                    /**< CfC增强群体 */
    SWARM_ENH_HYBRID                        /**< 混合模式 */
} SwarmEnhancedAlgorithm;

/**
 * @brief 分布式共识协议类型
 */
typedef enum {
    CONSENSUS_RAFT = 0,                     /**< RAFT共识（简化版） */
    CONSENSUS_PAXOS,                        /**< Paxos共识（简化版） */
    CONSENSUS_PBFT,                         /**< 拜占庭容错 */
    CONSENSUS_GOSSIP,                       /**< Gossip协议 */
    CONSENSUS_CFC_LIQUID,                   /**< CfC液态共识 */
    CONSENSUS_HYBRID                        /**< 混合共识 */
} ConsensusProtocol;

/**
 * @brief 节点状态（分布式一致性）
 */
typedef enum {
    NODE_STATE_FOLLOWER = 0,                /**< 跟随者 */
    NODE_STATE_CANDIDATE,                   /**< 候选者 */
    NODE_STATE_LEADER,                      /**< 领导者 */
    NODE_STATE_OBSERVER,                    /**< 观察者 */
    NODE_STATE_BYZANTINE,                   /**< 拜占庭节点 */
    NODE_STATE_OFFLINE                      /**< 离线 */
} NodeState;

/**
 * @brief ACO信息素更新策略（A06.4.1）
 */
typedef enum {
    ACO_PHEROMONE_ANT_CYCLE = 0,            /**< 蚂蚁周期更新 */
    ACO_PHEROMONE_ANT_QUANTITY,             /**< 蚂蚁数量更新 */
    ACO_PHEROMONE_ANT_DENSITY,              /**< 蚂蚁密度更新 */
    ACO_PHEROMONE_ADAPTIVE,                 /**< 自适应更新 */
    ACO_PHEROMONE_CFC_LIQUID                /**< CfC液态更新 */
} ACOPheromoneUpdate;

/**
 * @brief ACO增强配置
 */
typedef struct {
    int num_ants;                           /**< 蚂蚁数量 */
    float alpha;                            /**< 信息素重要性因子 */
    float beta;                             /**< 启发式信息重要性因子 */
    float evaporation_rate;                 /**< 信息素蒸发率ρ */
    float initial_pheromone;                /**< 初始信息素浓度 */
    ACOPheromoneUpdate update_strategy;     /**< 更新策略 */
    float elite_weight;                     /**< 精英蚂蚁权重 */
    float cfc_tau;                          /**< CfC时间常数 */
    float cfc_dt;                           /**< CfC步长 */
    int cfc_steps;                          /**< CfC积分步数 */
    int max_iterations;                     /**< 最大迭代次数 */
    int enable_adaptive_params;             /**< 启用自适应参数 */
    int enable_local_search;                /**< 启用局部搜索 */
} ACOEnhancedConfig;

/**
 * @brief ABC增强配置（A06.4.2）
 */
typedef struct {
    int colony_size;                        /**< 蜂群大小 */
    int food_sources;                       /**< 蜜源数 */
    int max_cycles;                         /**< 最大循环数 */
    float limit;                            /**< 放弃蜜源限制 */
    float scout_probability;                /**< 侦查蜂概率 */
    float cfc_tau;                          /**< CfC时间常数 */
    float cfc_dt;                           /**< CfC步长 */
    int cfc_steps;                          /**< CfC积分步数 */
    int enable_adaptive_search;             /**< 启用自适应搜索半径 */
    int enable_multi_objective;             /**< 启用多目标优化 */
} ABCEnhancedConfig;

/**
 * @brief 分布式共识节点配置（A06.4.3）
 */
typedef struct {
    int node_id;                            /**< 节点ID */
    NodeState initial_state;                /**< 初始状态 */
    float heartbeat_interval_ms;            /**< 心跳间隔(毫秒) */
    float election_timeout_min_ms;          /**< 选举超时最小值(毫秒) */
    float election_timeout_max_ms;          /**< 选举超时最大值(毫秒) */
    int vote_for;                           /**< 投票目标(-1=未投票) */
    long current_term;                      /**< 当前任期 */
    int is_byzantine;                       /**< 是否为拜占庭节点（测试用） */
} ConsensusNodeConfig;

/**
 * @brief 分布式共识日志条目
 */
typedef struct {
    long index;                             /**< 日志索引 */
    long term;                              /**< 日志任期 */
    int command_type;                       /**< 命令类型 */
    char command_data[SWARM_ENH_MAX_COMMAND_SIZE]; /**< 命令数据 */
    size_t command_size;                    /**< 命令大小 */
    int is_committed;                       /**< 是否已提交 */
} ConsensusLogEntry;

/**
 * @brief 群体液体通信消息
 */
typedef struct {
    int sender_id;                          /**< 发送者ID */
    int receiver_id;                        /**< 接收者ID(-1=广播) */
    long sequence_number;                   /**< 序列号 */
    int message_type;                       /**< 消息类型 */
    float* state_vector;                     /**< 状态向量 */
    size_t state_dim;                       /**< 状态维度 */
    float* cfc_hidden;                       /**< CfC隐状态 */
    size_t hidden_dim;                      /**< 隐状态维度 */
    float timestamp;                        /**< 时间戳 */
    int priority;                           /**< 优先级(0~255) */
    int is_ack_required;                    /**< 是否需要确认 */
} LiquidMessage;

/**
 * @brief 群体自愈配置
 */
typedef struct {
    int heartbeat_timeout_ms;               /**< 心跳超时(毫秒) */
    int max_failures_before_rebuild;        /**< 最大失败次数 */
    int enable_automatic_replacement;       /**< 启用自动替换 */
    int enable_task_redistribution;         /**< 启用任务重分配 */
    float cfc_recovery_tau;                 /**< CfC恢复时间常数 */
} SelfHealingConfig;

/**
 * @brief 增强群体引擎句柄
 */
typedef struct SwarmEnhancedEngine SwarmEnhancedEngine;

/* ============================================================================
 * A06.4.1 — ACO增强版API
 * ============================================================================ */

/**
 * @brief 创建增强ACO引擎
 *
 * @param config ACO配置
 * @return SwarmEnhancedEngine* 成功返回句柄，失败返回NULL
 */
SwarmEnhancedEngine* swarm_aco_enhanced_create(const ACOEnhancedConfig* config);

/**
 * @brief 销毁增强ACO引擎
 *
 * @param engine 引擎句柄
 */
void swarm_aco_enhanced_destroy(SwarmEnhancedEngine* engine);

/**
 * @brief 初始化ACO路径图
 *
 * @param engine 引擎句柄
 * @param num_nodes 节点数
 * @param distance_matrix 距离矩阵 [num_nodes][num_nodes]
 * @return int 成功返回0，失败返回-1
 */
int swarm_aco_init_graph(SwarmEnhancedEngine* engine,
                          int num_nodes,
                          const float* distance_matrix);

/**
 * @brief 执行一轮ACO迭代
 *
 * 蚂蚁构建路径：
 *   P_ij = [τ_ij]^α · [η_ij]^β / Σ [τ_ik]^α · [η_ik]^β
 * 其中τ_ij = 信息素，η_ij = 1/d_ij（启发式信息）
 *
 * CfC液态路径选择：
 *   P_cfc = softmax(W_cfc · h_ij / τ)
 * 融合选择：P = (1-λ)·P_aco + λ·P_cfc
 *
 * @param engine 引擎句柄
 * @param iteration 当前迭代次数
 * @return int 成功返回0，失败返回-1
 */
int swarm_aco_iterate(SwarmEnhancedEngine* engine, int iteration);

/**
 * @brief 获取ACO最佳路径
 *
 * @param engine 引擎句柄
 * @param path 输出路径节点数组
 * @param max_path_len 最大路径长度
 * @param total_distance 输出总距离
 * @return int 路径长度，失败返回-1
 */
int swarm_aco_get_best_path(SwarmEnhancedEngine* engine,
                             int* path, int max_path_len,
                             float* total_distance);

/**
 * @brief 获取ACO信息素矩阵
 *
 * @param engine 引擎句柄
 * @param pheromone_matrix 输出信息素矩阵 [num_nodes][num_nodes]
 * @param max_size 最大大小
 * @return int 成功返回0，失败返回-1
 */
int swarm_aco_get_pheromone(SwarmEnhancedEngine* engine,
                             float* pheromone_matrix, size_t max_size);

/* ============================================================================
 * A06.4.2 — ABC增强版API
 * ============================================================================ */

/**
 * @brief 创建增强ABC引擎
 *
 * @param config ABC配置
 * @return SwarmEnhancedEngine* 成功返回句柄，失败返回NULL
 */
SwarmEnhancedEngine* swarm_abc_enhanced_create(const ABCEnhancedConfig* config);

/**
 * @brief 初始化ABC蜜源
 *
 * @param engine 引擎句柄
 * @param dimensions 问题维度
 * @param lower_bound 下界数组 [dimensions]
 * @param upper_bound 上界数组 [dimensions]
 * @return int 成功返回0，失败返回-1
 */
int swarm_abc_init_sources(SwarmEnhancedEngine* engine,
                            int dimensions,
                            const float* lower_bound,
                            const float* upper_bound);

/**
 * @brief 执行一轮ABC迭代
 *
 * 雇佣蜂阶段：邻域搜索
 *   v_ij = x_ij + φ_ij * (x_ij - x_kj)
 *
 * 观察蜂阶段：轮盘赌选择
 *   P_i = fitness_i / Σ fitness_j
 *
 * 侦查蜂阶段：随机生成新蜜源
 *
 * CfC增强蜜源评估：
 *   quality_cfc = CFC_ODE(h_source, t)
 *   未来质量的液态预测
 *
 * @param engine 引擎句柄
 * @param cycle 当前循环数
 * @return int 成功返回0，失败返回-1
 */
int swarm_abc_iterate(SwarmEnhancedEngine* engine, int cycle);

/**
 * @brief 获取ABC最佳解
 *
 * @param engine 引擎句柄
 * @param solution 输出最佳解 [dimensions]
 * @param max_dim 最大维度
 * @param fitness 输出适应度
 * @return int 维度数，失败返回-1
 */
int swarm_abc_get_best_solution(SwarmEnhancedEngine* engine,
                                 float* solution, int max_dim,
                                 float* fitness);

/* ============================================================================
 * A06.4.3 — 分布式一致性API
 * ============================================================================ */

/**
 * @brief 创建分布式共识节点
 *
 * @param config 节点配置
 * @param protocol 共识协议
 * @return SwarmEnhancedEngine* 成功返回句柄，失败返回NULL
 */
SwarmEnhancedEngine* swarm_consensus_create_node(
    const ConsensusNodeConfig* config, ConsensusProtocol protocol);

/**
 * @brief 注册其他节点
 *
 * @param engine 引擎句柄
 * @param node_ids 其他节点ID数组
 * @param num_nodes 节点数
 * @return int 成功返回0，失败返回-1
 */
int swarm_consensus_register_peers(SwarmEnhancedEngine* engine,
                                    const int* node_ids, int num_nodes);

/**
 * @brief 发起选举（RAFT风格）
 *
 * 节点成为候选者，向其他节点发送投票请求：
 *   请求投票(RequestVote RPC)：
 *     term, candidateId, lastLogIndex, lastLogTerm
 *   投票条件：候选者的日志至少和自己一样新
 *   多数通过：获得N/2+1票
 *
 * @param engine 引擎句柄
 * @return int 成功(当选)返回1，失败返回0，错误返回-1
 */
int swarm_consensus_start_election(SwarmEnhancedEngine* engine);

/**
 * @brief 追加日志条目（领导者专用）
 *
 * @param engine 引擎句柄
 * @param entry 日志条目(命令)
 * @return int 成功返回0，失败返回-1
 */
int swarm_consensus_append_entry(SwarmEnhancedEngine* engine,
                                  const ConsensusLogEntry* entry);

/**
 * @brief 提交日志到状态机
 *
 * @param engine 引擎句柄
 * @param index 日志索引
 * @return int 成功返回0，失败返回-1
 */
int swarm_consensus_commit_entry(SwarmEnhancedEngine* engine, long index);

/**
 * @brief 获取当前领导者ID
 *
 * @param engine 引擎句柄
 * @return int 领导者ID，无领导者返回-1
 */
int swarm_consensus_get_leader(SwarmEnhancedEngine* engine);

/**
 * @brief 获取节点当前状态
 *
 * @param engine 引擎句柄
 * @return NodeState 节点状态
 */
NodeState swarm_consensus_get_state(SwarmEnhancedEngine* engine);

/* ============================================================================
 * A06.4.4 — 群体液态通信API
 * ============================================================================ */

/**
 * @brief 创建群体液态通信通道
 *
 * @param engine 父引擎句柄
 * @param num_nodes 节点数
 * @param state_dim 状态维度
 * @param hidden_dim CfC隐状态维度
 * @return int 成功返回通道ID，失败返回-1
 */
int swarm_liquid_comm_create(SwarmEnhancedEngine* engine,
                              int num_nodes, int state_dim, int hidden_dim);

/**
 * @brief 发送液态消息
 *
 * CfC ODE消息编码：
 *   h_msg = CFC_ODE_encode(hidden, state, t)
 *   消息通过液态传播同步接收方状态
 *
 * @param engine 引擎句柄
 * @param channel_id 通道ID
 * @param message 消息内容
 * @return int 成功返回0，失败返回-1
 */
int swarm_liquid_comm_send(SwarmEnhancedEngine* engine,
                            int channel_id, const LiquidMessage* message);

/**
 * @brief 接收液态消息
 *
 * @param engine 引擎句柄
 * @param channel_id 通道ID
 * @param message 输出消息缓冲区
 * @param block 是否阻塞(0=非阻塞)
 * @return int 成功返回1，无消息返回0，失败返回-1
 */
int swarm_liquid_comm_receive(SwarmEnhancedEngine* engine,
                               int channel_id, LiquidMessage* message, int block);

/**
 * @brief 执行群体状态同步（CfC ODE平均共识）
 *
 * 所有节点的CfC隐状态通过液态通信同步到共同吸引子：
 *   dh_i/dt = -h_i/τ + Σ_j W_ij · f_cfc(h_i, h_j, m_ij)
 * 其中m_ij是节点i和j之间的消息
 *
 * @param engine 引擎句柄
 * @param channel_id 通道ID
 * @param sync_steps 同步步数
 * @return int 成功返回0，失败返回-1
 */
int swarm_liquid_comm_sync(SwarmEnhancedEngine* engine,
                            int channel_id, int sync_steps);

/* ============================================================================
 * A06.4.5 — 群体自愈与重组API
 * ============================================================================ */

/**
 * @brief 配置群体自愈
 *
 * @param engine 引擎句柄
 * @param config 自愈配置
 * @return int 成功返回0，失败返回-1
 */
int swarm_self_healing_configure(SwarmEnhancedEngine* engine,
                                  const SelfHealingConfig* config);

/**
 * @brief 检测故障节点
 *
 * 心跳超时检测 + CfC异常检测：
 *   如果是心跳超时：标记为离线
 *   如果CfC异常分数高：标记为拜占庭节点
 *
 * @param engine 引擎句柄
 * @param failed_nodes 输出故障节点ID数组
 * @param max_failures 最大故障数
 * @return int 故障节点数
 */
int swarm_self_healing_detect_failures(SwarmEnhancedEngine* engine,
                                        int* failed_nodes, int max_failures);

/**
 * @brief 替换故障节点
 *
 * 1. 从备用节点池中选择替换
 * 2. 用CfC恢复被替换节点的状态
 *    h_new = CFC_init(h_backup, peers)
 * 3. 重分配故障节点的任务
 *
 * @param engine 引擎句柄
 * @param failed_node_id 故障节点ID
 * @param replacement_id 替换节点ID
 * @return int 成功返回0，失败返回-1
 */
int swarm_self_healing_replace_node(SwarmEnhancedEngine* engine,
                                     int failed_node_id, int replacement_id);

/**
 * @brief 群体任务重分配
 *
 * 当节点故障后，自动重分配其任务给其他节点：
 *   使用匈牙利算法或贪心算法最小化最大负载
 *
 * @param engine 引擎句柄
 * @param failed_node_id 故障节点ID
 * @return int 成功返回0，失败返回-1
 */
int swarm_self_healing_redistribute_tasks(SwarmEnhancedEngine* engine,
                                           int failed_node_id);

/* ============================================================================
 * 模型管理
 * ============================================================================ */

/**
 * @brief 保存增强群体模型
 *
 * @param engine 引擎句柄
 * @param algorithm 算法类型
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int swarm_enhanced_save(const SwarmEnhancedEngine* engine,
                         SwarmEnhancedAlgorithm algorithm,
                         const char* filepath);

/**
 * @brief 加载增强群体模型
 *
 * @param algorithm 算法类型
 * @param filepath 文件路径
 * @return SwarmEnhancedEngine* 成功返回句柄，失败返回NULL
 */
SwarmEnhancedEngine* swarm_enhanced_load(SwarmEnhancedAlgorithm algorithm,
                                          const char* filepath);

/**
 * @brief 获取默认ACO增强配置
 *
 * @return ACOEnhancedConfig 默认配置
 */
ACOEnhancedConfig swarm_aco_default_config(void);

/**
 * @brief 获取默认ABC增强配置
 *
 * @return ABCEnhancedConfig 默认配置
 */
ABCEnhancedConfig swarm_abc_default_config(void);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SWARM_ENHANCED_H */
