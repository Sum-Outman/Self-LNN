/**
 * @file multi_agent.h
 * @brief 多智能体协作框架
 * 
 * 多智能体系统，支持智能体通信、协作、协调和集体决策。
 * 提供集中式、分散式、混合式架构，支持多种协作策略。
 */

#ifndef SELFLNN_MULTI_AGENT_H
#define SELFLNN_MULTI_AGENT_H

#include "selflnn/core/cfc_network.h"
#include "selflnn/learning/learning.h"
#include "selflnn/core/common.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 兼容性定义：NeuralNetwork 是 CfCNetwork 的别名
/* 兼容性定义：NeuralNetwork 是 CfCNetwork 的别名 */
#ifndef SELFLNN_NEURAL_NETWORK_DEFINED
#define SELFLNN_NEURAL_NETWORK_DEFINED
typedef struct CfCNetwork NeuralNetwork;
#endif

/**
 * @brief 智能体类型
 */
typedef enum {
    AGENT_TYPE_SIMPLE = 0,          /**< 简单反应型智能体 */
    AGENT_TYPE_LEARNING,            /**< 学习型智能体 */
    AGENT_TYPE_ADAPTIVE,            /**< 自适应智能体 */
    AGENT_TYPE_COOPERATIVE,         /**< 协作型智能体 */
    AGENT_TYPE_COMPETITIVE,         /**< 竞争型智能体 */
    AGENT_TYPE_HYBRID,              /**< 混合型智能体 */
    AGENT_TYPE_EXPLORER,            /**< 探索型智能体：探索未知环境 */
    AGENT_TYPE_EXPLOITER,           /**< 利用型智能体：利用已知知识 */
    AGENT_TYPE_COLLABORATOR         /**< 协作型智能体：专注于协作任务 */
} AgentType;

/**
 * @brief 动作类型
 */
typedef enum {
    ACTION_TYPE_NONE = 0,           /**< 无动作 */
    ACTION_TYPE_EXPLORE,            /**< 探索动作：探索环境 */
    ACTION_TYPE_EXPLOIT,            /**< 利用动作：利用已知信息 */
    ACTION_TYPE_COLLABORATE,        /**< 协作动作：与其他智能体协作 */
    ACTION_TYPE_COMMUNICATE,        /**< 通信动作：发送消息 */
    ACTION_TYPE_LEARN,              /**< 学习动作：更新知识 */
    ACTION_TYPE_REST,               /**< 休息动作：恢复能量 */
    ACTION_TYPE_OBSERVE,            /**< 观察动作：收集信息 */
    ACTION_TYPE_MOVE,               /**< 移动动作：物理移动 */
    ACTION_TYPE_GRAB,               /**< 抓取动作：抓取物体 */
    ACTION_TYPE_RELEASE,            /**< 释放动作：释放物体 */
    ACTION_TYPE_COMMAND,            /**< 命令动作：发送命令 */
    ACTION_TYPE_ADAPT,              /**< 适应动作：自适应调整 */
    ACTION_TYPE_EVOLVE              /**< 进化动作：自我进化 */
} ActionType;

/**
 * @brief 智能体角色
 */
typedef enum {
    AGENT_ROLE_WORKER = 0,          /**< 工作者：执行具体任务 */
    AGENT_ROLE_COORDINATOR,         /**< 协调者：协调其他智能体 */
    AGENT_ROLE_MONITOR,             /**< 监控者：监控系统状态 */
    AGENT_ROLE_NEGOTIATOR,          /**< 协商者：处理冲突和协商 */
    AGENT_ROLE_LEADER,              /**< 领导者：做出最终决策 */
    AGENT_ROLE_SPECIALIST           /**< 专家：特定领域专家 */
} AgentRole;

/**
 * @brief 协作策略
 */
typedef enum {
    COOPERATION_NONE = 0,           /**< 无协作 */
    COOPERATION_SHARED_GOAL,        /**< 共享目标 */
    COOPERATION_TASK_ALLOCATION,    /**< 任务分配 */
    COOPERATION_CONSENSUS,          /**< 共识达成 */
    COOPERATION_NEGOTIATION,        /**< 协商 */
    COOPERATION_COALITION,          /**< 联盟形成 */
    COOPERATION_MARKET_BASED        /**< 基于市场的协作 */
} CooperationStrategy;

/**
 * @brief 通信协议
 */
typedef enum {
    COMM_PROTOCOL_DIRECT = 0,       /**< 直接通信 */
    COMM_PROTOCOL_BROADCAST,        /**< 广播通信 */
    COMM_PROTOCOL_PUBLISH_SUBSCRIBE, /**< 发布订阅 */
    COMM_PROTOCOL_BLACKBOARD,       /**< 黑板系统 */
    COMM_PROTOCOL_MESSAGE_PASSING,  /**< 消息传递 */
    COMM_PROTOCOL_FEDERATED,        /**< 联邦学习式通信 */
    PROTOCOL_BROADCAST = COMM_PROTOCOL_BROADCAST, /**< 兼容性定义 */
    PROTOCOL_DIRECT = COMM_PROTOCOL_DIRECT        /**< 兼容性定义 */
} CommunicationProtocol;

/* 日志级别使用common.h中的定义 */
/* 兼容性宏：LOG_LEVEL_WARN 映射到 LOG_LEVEL_WARNING */
#ifndef LOG_LEVEL_WARN
#define LOG_LEVEL_WARN LOG_LEVEL_WARNING
#endif

/**
 * @brief 共识算法
 */
typedef enum {
    CONSENSUS_AVERAGE = 0,          /**< 平均值共识 */
    CONSENSUS_MAJORITY,             /**< 多数投票 */
    CONSENSUS_WEIGHTED_AVERAGE,     /**< 加权平均 */
    CONSENSUS_MAXIMUM,              /**< 最大值 */
    CONSENSUS_MINIMUM,              /**< 最小值 */
    CONSENSUS_MEDIAN,               /**< 中位数 */
    CONSENSUS_WEIGHTED = CONSENSUS_WEIGHTED_AVERAGE  /**< 加权共识（兼容性别名） */
} ConsensusAlgorithm;

/**
 * @brief 智能体状态
 */
typedef struct {
    int agent_id;                   /**< 智能体ID */
    AgentType agent_type;           /**< 智能体类型 */
    AgentRole agent_role;           /**< 智能体角色 */
    
    // 能力
    float capability_level;         /**< 能力水平（0-1） */
    float expertise_level;          /**< 专业水平（0-1） */
    float trustworthiness;          /**< 可信度（0-1） */
    float trust_level;              /**< 信任级别（0-1） */
    
    // 状态
    int busy;                       /**< 是否忙碌 */
    int available;                  /**< 是否可用 */
    float energy_level;             /**< 能量水平（0-1） */
    
    // 性能
    float success_rate;             /**< 成功率 */
    float efficiency;               /**< 效率 */
    float collaboration_score;      /**< 协作得分 */
    float capability_learning;      /**< 能力学习率 */
    float adaptability;             /**< 适应性 */
    float robustness;               /**< 鲁棒性 */
} AgentState;

/**
 * @brief 个体性能统计
 */
typedef struct {
    int agent_id;                   /**< 智能体ID */
    float cumulative_reward;        /**< 累积奖励 */
    float success_rate;             /**< 成功率 */
    int task_completed;             /**< 完成任务数 */
    float average_decision_quality; /**< 平均决策质量 */
    float learning_progress;        /**< 学习进度 */
    float capability_level;         /**< 能力水平 */
    float collaboration_score;      /**< 协作得分 */
} IndividualPerformance;

/**
 * @brief 系统性能统计
 */
typedef struct {
    int total_tasks;                /**< 总任务数 */
    int completed_tasks;            /**< 完成任务数 */
    int failed_tasks;               /**< 失败任务数 */
    float total_reward;             /**< 总奖励 */
    float average_reward;           /**< 平均奖励 */
    float max_reward;               /**< 最大奖励 */
    float min_reward;               /**< 最小奖励 */
    float total_decision_time;      /**< 总决策时间（毫秒） */
    float average_decision_time;    /**< 平均决策时间（毫秒） */
    float total_communication_time; /**< 总通信时间（毫秒） */
    float average_communication_time; /**< 平均通信时间（毫秒） */
    float system_utilization;       /**< 系统利用率（0-1） */
    float load_balance;             /**< 负载均衡度（0-1） */
    float fairness_index;           /**< 公平性指数（0-1） */
    
    // 性能指标
    IndividualPerformance* individual_performance;  /**< 个体性能数组 */
    float average_success_rate;     /**< 平均成功率 */
    float collaboration_efficiency; /**< 协作效率 */
    float communication_efficiency; /**< 通信效率 */
    
    // 统计计数
    int total_decisions;            /**< 总决策数 */
    int total_tasks_completed;      /**< 总完成任务数 */
    int total_tasks_created;        /**< 总创建任务数 */
    
    // 奖励和评估
    float global_reward;            /**< 全局奖励 */
    float evaluation_time;          /**< 评估时间（毫秒） */
    
    // 时间统计
    float total_execution_time;     /**< 总执行时间（毫秒） */
    float average_execution_time;   /**< 平均执行时间（毫秒） */
    float max_execution_time;       /**< 最大执行时间（毫秒） */
    float min_execution_time;       /**< 最小执行时间（毫秒） */
} SystemPerformance;

/**
 * @brief 智能体配置
 */
typedef struct {
    char* agent_id;                 /**< 智能体ID字符串 */
    AgentType type;                 /**< 智能体类型 */
    AgentRole role;                 /**< 智能体角色 */
    
    // 能力配置
    float initial_capability;       /**< 初始能力 */
    float learning_rate;            /**< 学习率 */
    float exploration_rate;         /**< 探索率 */
    
    // 通信配置
    CommunicationProtocol comm_protocol; /**< 通信协议 */
    float comm_range;               /**< 通信范围 */
    float comm_frequency;           /**< 通信频率 */
    
    // 协作配置
    CooperationStrategy strategy;   /**< 协作策略 */
    float cooperation_weight;       /**< 协作权重 */
    float competition_weight;       /**< 竞争权重 */
    
    // 资源限制
    float max_energy;               /**< 最大能量 */
    float max_memory;               /**< 最大内存 */
    float max_computation;          /**< 最大计算能力 */
    
    // 学习参数
    float discount_factor;          /**< 折扣因子（用于强化学习） */
    int memory_size;                /**< 记忆缓冲区大小 */
    int batch_size;                 /**< 批次大小 */
    
    // 优化参数
    float entropy_weight;           /**< 熵权重（用于探索） */
    float value_weight;             /**< 价值函数权重 */
    float policy_weight;            /**< 策略权重 */
    
    // 通信参数
    float communication_range;      /**< 通信范围 */
    int max_messages;               /**< 最大消息数 */
    int message_ttl;                /**< 消息生存时间 */
    
    // 更新参数
    int update_frequency;           /**< 更新频率（步数） */
    int sync_frequency;             /**< 同步频率（步数） */
    int evaluation_frequency;       /**< 评估频率（步数） */
    
    // 自适应配置
    int enable_adaptation;          /**< 启用自适应 */
    int enable_self_improvement;    /**< 启用自我改进 */
    int enable_transfer_learning;   /**< 启用迁移学习 */
    
    // 协作配置
    float collaboration_threshold;  /**< 协作阈值 */
    float trust_update_rate;        /**< 信任更新率 */
    
    // 功能开关
    int enable_learning;            /**< 启用学习 */
    int enable_communication;       /**< 启用通信 */
    int enable_collaboration;       /**< 启用协作 */
    
    // 性能参数
    float performance_weight;       /**< 性能权重 */
    float stability_weight;         /**< 稳定性权重 */
    float adaptability_weight;      /**< 适应性权重 */
} AgentConfig;

/**
 * @brief 智能体
 */
typedef struct Agent Agent;

/**
 * @brief 多智能体系统配置
 */
typedef struct {
    int agent_count;                /**< 智能体数量 */
    AgentConfig* agent_configs;     /**< 智能体配置数组 */
    
    // 系统架构
    int centralized;                /**< 是否集中式 */
    int hierarchical;               /**< 是否分层 */
    int distributed;                /**< 是否分布式 */
    
    // 协作设置
    CooperationStrategy global_strategy; /**< 全局协作策略 */
    float collaboration_threshold;  /**< 协作阈值 */
    float consensus_threshold;      /**< 共识阈值 */
    
    // 通信设置
    CommunicationProtocol global_protocol; /**< 全局通信协议 */
    float network_density;          /**< 网络密度 */
    float communication_delay;      /**< 通信延迟 */
    
    // 学习设置
    int enable_collective_learning; /**< 启用集体学习 */
    int enable_knowledge_sharing;   /**< 启用知识共享 */
    int enable_experience_replay;   /**< 启用经验回放 */
    
    // 优化目标
    float goal_global_reward;       /**< 全局奖励权重 */
    float goal_individual_reward;   /**< 个体奖励权重 */
    float goal_fairness;            /**< 公平性权重 */
    
    // 系统参数
    int decentralized;              /**< 是否去中心化 */
    int communication_protocol;     /**< 通信协议（兼容性字段） */
    ConsensusAlgorithm consensus_algorithm; /**< 共识算法 */
    int max_tasks;                  /**< 最大任务数 */
    int max_messages;               /**< 最大消息数 */
    int knowledge_size;             /**< 知识库大小 */
    int synchronization_interval;   /**< 同步间隔 */
    int consensus_rounds;           /**< 共识轮数 */
    
    // 学习配置
    int enable_global_learning;     /**< 启用全局学习 */
    int enable_local_learning;      /**< 启用本地学习 */
    int enable_consensus;           /**< 启用共识 */
    float learning_rate;            /**< 学习率 */
    
    // 自适应配置
    int enable_adaptation;          /**< 启用自适应 */
    int enable_self_organization;   /**< 启用自组织 */
    int enable_evolution;           /**< 启用进化 */
    
    // 权重配置
    float collaboration_weight;     /**< 协作权重 */
    float communication_weight;     /**< 通信权重 */
    float exploration_weight;       /**< 探索权重 */
    
    // 系统配置
    int evaluation_interval;        /**< 评估间隔 */
    LogLevel logging_level;         /**< 日志级别 */
    
    // 监控配置
    int enable_monitoring;          /**< 启用监控 */
    int enable_performance_logging; /**< 启用性能日志 */
    int enable_error_reporting;     /**< 启用错误报告 */
} MultiAgentConfig;

/**
 * @brief 多智能体系统
 */
typedef struct MultiAgentSystem MultiAgentSystem;

/**
 * @brief 智能体消息
 */
typedef struct {
    int sender_id;                  /**< 发送者ID */
    int receiver_id;                /**< 接收者ID */
    char* message_type;             /**< 消息类型 */
    void* message_data;             /**< 消息数据 */
    size_t data_size;               /**< 数据大小 */
    char* content;                  /**< 消息文本内容 */
    size_t content_size;            /**< 内容大小（字节数） */
    float timestamp;                /**< 时间戳 */
    float priority;                 /**< 优先级 */
    int ttl;                        /**< 生存时间（time to live） */
    int delivered;                  /**< 是否已送达 */
    int read;                       /**< 是否已读取 */
    int response_required;          /**< 是否需要响应 */
    int response_received;          /**< 是否收到响应 */
} AgentMessage;

/**
 * @brief 智能体动作
 */
typedef struct {
    int action_type;                /**< 动作类型 */
    float* action_values;           /**< 动作值数组 */
    size_t values_size;             /**< 值数组大小 */
    float confidence;               /**< 置信度（0-1） */
    float expected_reward;          /**< 预期奖励 */
    int target_agent_id;            /**< 目标智能体ID（-1表示无目标） */
    int task_id;                    /**< 关联任务ID（-1表示无关联） */
    float* action_params;           /**< 动作参数数组 */
    size_t params_size;             /**< 参数数组大小 */
} AgentAction;

/**
 * @brief 智能体经验
 */
typedef struct {
    float* state_observation;       /**< 状态观察值 */
    float* next_state_observation;  /**< 下一状态观察值 */
    AgentAction* action;            /**< 执行的动作 */
    float reward;                   /**< 获得的奖励 */
    float state_value;              /**< 状态价值 */
    float next_state_value;         /**< 下一状态价值 */
    int done;                       /**< 是否结束 */
    size_t obs_size;                /**< 观察值大小 */
} AgentExperience;

/**
 * @brief 任务类型
 */
typedef enum {
    TASK_TYPE_DECISION = 0,         /**< 决策任务 */
    TASK_TYPE_PLANNING,             /**< 规划任务 */
    TASK_TYPE_EXECUTION,            /**< 执行任务 */
    TASK_TYPE_MONITORING,           /**< 监控任务 */
    TASK_TYPE_LEARNING,             /**< 学习任务 */
    TASK_TYPE_COLLABORATION,        /**< 协作任务 */
    TASK_TYPE_EXPLORATION,          /**< 探索任务 */
    TASK_TYPE_EXPLOITATION,         /**< 利用任务 */
    TASK_TYPE_PERCEPTION,           /**< 感知任务 */
    TASK_TYPE_ACTION,               /**< 动作任务 */
    TASK_TYPE_MEMORY,               /**< 记忆任务 */
    TASK_TYPE_REASONING,            /**< 推理任务 */
    TASK_TYPE_ADAPTATION            /**< 适应任务 */
} TaskType;

/**
 * @brief 任务状态
 */
typedef enum {
    TASK_STATUS_PENDING = 0,        /**< 待处理 */
    TASK_STATUS_ASSIGNED,           /**< 已分配 */
    TASK_STATUS_IN_PROGRESS,        /**< 进行中 */
    TASK_STATUS_COMPLETED,          /**< 已完成 */
    TASK_STATUS_FAILED,             /**< 失败 */
    TASK_STATUS_CANCELLED           /**< 取消 */
} TaskStatus;

/**
 * @brief 协作类型
 */
typedef enum {
    COLLABORATION_SEQUENTIAL = 0,   /**< 顺序协作 */
    COLLABORATION_PARALLEL,         /**< 并行协作 */
    COLLABORATION_PIPELINE,         /**< 流水线协作 */
    COLLABORATION_HIERARCHICAL,     /**< 分层协作 */
    COLLABORATION_DECENTRALIZED     /**< 去中心化协作 */
} CollaborationType;

/**
 * @brief 协作任务
 */
typedef struct {
    char* task_id;                  /**< 任务ID */
    int required_agents;            /**< 所需智能体数量 */
    float difficulty;               /**< 任务难度 */
    float urgency;                  /**< 紧急程度 */
    float reward;                   /**< 任务奖励 */
    
    // 任务类型和状态
    TaskType task_type;             /**< 任务类型 */
    TaskStatus task_status;         /**< 任务状态 */
    CollaborationType collaboration_type; /**< 协作类型 */
    
    // 任务参数和数据
    void* input_data;               /**< 输入数据 */
    void* output_data;              /**< 输出数据 */
    float* task_parameters;         /**< 任务参数数组 */
    size_t params_size;             /**< 参数数组大小 */
    
    // 智能体分配
    int* assigned_agents_list;      /**< 分配到的智能体ID列表（动态分配） */
    int assigned_agents;            /**< 已分配智能体数 */
    int assigned_agent;             /**< 单个分配智能体索引 */
    int completed_agents;           /**< 已完成智能体数 */
    
    // 时间管理
    time_t creation_time;           /**< 创建时间 */
    time_t start_time;              /**< 开始时间 */
    time_t completion_time;         /**< 完成时间 */
    time_t deadline;                /**< 截止时间 */
    
    // 进度和质量指标
    float completion_ratio;         /**< 完成比例（0-1） */
    float priority;                 /**< 优先级（0-1） */
    float quality;                  /**< 质量评分（0-1） */
    float efficiency;               /**< 效率评分（0-1） */
    float collaboration_score;      /**< 协作评分（0-1） */
    
    // 任务执行统计
    float energy_consumed;          /**< 累计能量消耗 */
    float total_contributions;      /**< 总贡献度 */
    int execution_steps;            /**< 执行步骤计数 */
    
    // 状态标志
    int completed;                  /**< 是否完成 */
} CollaborativeTask;

/**
 * @brief 创建多智能体系统
 * 
 * @param config 系统配置
 * @return MultiAgentSystem* 多智能体系统指针，失败返回NULL
 */
MultiAgentSystem* multi_agent_system_create(const MultiAgentConfig* config);

/**
 * @brief 销毁多智能体系统
 * 
 * @param system 多智能体系统
 */
void multi_agent_system_destroy(MultiAgentSystem* system);

/**
 * @brief 添加智能体到系统
 * 
 * @param system 多智能体系统
 * @param agent_config 智能体配置
 * @return int 智能体ID，失败返回-1
 */
int multi_agent_add_agent(MultiAgentSystem* system, const AgentConfig* agent_config);

/**
 * @brief 从系统移除智能体
 * 
 * @param system 多智能体系统
 * @param agent_id 智能体ID
 * @return int 成功返回0，失败返回-1
 */
int multi_agent_remove_agent(MultiAgentSystem* system, int agent_id);

/**
 * @brief 执行协作任务
 * 
 * @param system 多智能体系统
 * @param task 协作任务
 * @return int 成功返回0，失败返回-1
 */
int multi_agent_execute_task(MultiAgentSystem* system, CollaborativeTask* task);

/**
 * @brief 智能体间通信
 * 
 * @param system 多智能体系统
 * @param message 消息
 * @return int 成功返回0，失败返回-1
 */
int multi_agent_send_message(MultiAgentSystem* system, const AgentMessage* message);

/**
 * @brief 接收智能体消息
 * 
 * @param system 多智能体系统
 * @param agent_id 智能体ID
 * @param message 消息输出
 * @return int 消息数量，失败返回-1
 */
int multi_agent_receive_messages(MultiAgentSystem* system, int agent_id, AgentMessage** messages);

/**
 * @brief 形成协作联盟
 * 
 * @param system 多智能体系统
 * @param task 任务
 * @param agent_ids 智能体ID输出数组
 * @param max_agents 最大智能体数
 * @return int 联盟大小，失败返回-1
 */
int multi_agent_form_coalition(MultiAgentSystem* system, const CollaborativeTask* task,
                              int* agent_ids, int max_agents);

/**
 * @brief 达成共识
 * 
 * @param system 多智能体系统
 * @param agent_ids 智能体ID数组
 * @param agent_count 智能体数量
 * @param proposal 提议
 * @param result 结果输出
 * @return int 成功返回0，失败返回-1
 */
int multi_agent_reach_consensus(MultiAgentSystem* system, const int* agent_ids,
                               int agent_count, const void* proposal, void* result);

/**
 * @brief 分配任务
 * 
 * @param system 多智能体系统
 * @param task 任务
 * @param agent_ids 智能体ID输出数组
 * @return int 分配的智能体数量，失败返回-1
 */
int multi_agent_allocate_task(MultiAgentSystem* system, CollaborativeTask* task,
                             int* agent_ids);

/**
 * @brief 更新智能体状态
 * 
 * @param system 多智能体系统
 * @param agent_id 智能体ID
 * @param new_state 新状态
 * @return int 成功返回0，失败返回-1
 */
int multi_agent_update_state(MultiAgentSystem* system, int agent_id,
                            const AgentState* new_state);

/**
 * @brief 获取智能体状态
 * 
 * @param system 多智能体系统
 * @param agent_id 智能体ID
 * @param state 状态输出
 * @return int 成功返回0，失败返回-1
 */
int multi_agent_get_state(MultiAgentSystem* system, int agent_id, AgentState* state);

/**
 * @brief 评估系统性能
 * 
 * @param system 多智能体系统
 * @param metrics 指标输出数组
 * @param metric_count 指标数量
 * @return int 成功返回0，失败返回-1
 */
int multi_agent_evaluate_performance(MultiAgentSystem* system, float* metrics,
                                    int metric_count);

/**
 * @brief 集体学习
 * 
 * @param system 多智能体系统
 * @param learning_data 学习数据
 * @param data_size 数据大小
 * @return int 成功返回0，失败返回-1
 */
int multi_agent_collective_learning(MultiAgentSystem* system, const void* learning_data,
                                   size_t data_size);

/**
 * @brief 知识共享
 * 
 * @param system 多智能体系统
 * @param source_agent 源智能体ID
 * @param target_agents 目标智能体ID数组
 * @param target_count 目标数量
 * @param knowledge_data 知识数据
 * @param data_size 数据大小
 * @return int 成功返回0，失败返回-1
 */
int multi_agent_share_knowledge(MultiAgentSystem* system, int source_agent,
                               const int* target_agents, int target_count,
                               const void* knowledge_data, size_t data_size);

/**
 * @brief 获取默认多智能体配置
 * 
 * @param config 配置输出
 */
void multi_agent_default_config(MultiAgentConfig* config);

/**
 * @brief 获取默认智能体配置
 * 
 * @param config 配置输出
 */
void agent_default_config(AgentConfig* config);

/**
 * @brief 创建协作任务
 * 
 * @param task_id 任务ID
 * @param required_agents 所需智能体数量
 * @return CollaborativeTask* 任务指针，失败返回NULL
 */
CollaborativeTask* collaborative_task_create(const char* task_id, int required_agents);

/**
 * @brief 销毁协作任务
 * 
 * @param task 协作任务
 */
void collaborative_task_destroy(CollaborativeTask* task);

/**
 * @brief 创建智能体消息
 * 
 * @param sender_id 发送者ID
 * @param receiver_id 接收者ID
 * @param message_type 消息类型
 * @return AgentMessage* 消息指针，失败返回NULL
 */
AgentMessage* agent_message_create(int sender_id, int receiver_id, const char* message_type);

/**
 * @brief 销毁智能体消息
 * 
 * @param message 智能体消息
 */
void agent_message_destroy(AgentMessage* message);

/* ============================================================================
 * CfC多智能体强化学习算法
 * =========================================================================== */

/**
 * @brief CfC多智能体算法类型
 */
typedef enum {
    MA_CFC_ALGORITHM_MADDPG = 0,  /**< CfC-MADDPG：多智能体DDPG */
    MA_CFC_ALGORITHM_QMIX,        /**< CfC-QMIX：值函数分解QMIX */
    MA_CFC_ALGORITHM_VDN,         /**< CfC-VDN：值函数分解VDN */
    MA_CFC_ALGORITHM_COMA         /**< CfC-COMA：反事实多智能体策略梯度 */
} MACfcAlgorithm;

/**
 * @brief CfC-MADDPG配置
 */
typedef struct {
    int num_agents;                /**< 智能体数量 */
    int state_dim;                 /**< 每个智能体的状态维度 */
    int action_dim;                /**< 每个智能体的动作维度 */
    int hidden_size;               /**< 隐藏层大小 */
    float actor_lr;                /**< 演员学习率 */
    float critic_lr;               /**< 评论家学习率 */
    float gamma;                   /**< 折扣因子 */
    float tau;                     /**< 软更新系数 */
    float exploration_noise;       /**< 探索噪声标准差 */
    float noise_clip;              /**< 噪声裁剪范围 */
    int batch_size;                /**< 批次大小 */
    int buffer_capacity;           /**< 经验回放容量 */
    int cfc_ode_solver;            /**< CfC ODE求解器类型 */
    float cfc_time_constant;       /**< CfC时间常数 */
} MACfcMADDPGConfig;

/**
 * @brief CfC-QMIX配置
 */
typedef struct {
    int num_agents;                /**< 智能体数量 */
    int state_dim;                 /**< 每个智能体的状态维度 */
    int action_dim;                /**< 每个智能体的动作维度 */
    int hidden_size;               /**< 隐藏层大小 */
    int mixing_hidden_size;        /**< 混合网络隐藏层大小 */
    float lr;                      /**< 学习率 */
    float gamma;                   /**< 折扣因子 */
    float tau;                     /**< 目标网络软更新系数 */
    int target_update_interval;    /**< 目标网络硬更新间隔 */
    int batch_size;                /**< 批次大小 */
    int buffer_capacity;           /**< 经验回放容量 */
    int cfc_ode_solver;            /**< CfC ODE求解器类型 */
    float cfc_time_constant;       /**< CfC时间常数 */
    int double_q;                  /**< 是否使用Double Q-learning */
} MACfcQMIXConfig;

/**
 * @brief CfC-VDN配置
 */
typedef struct {
    int num_agents;                /**< 智能体数量 */
    int state_dim;                 /**< 每个智能体的状态维度 */
    int action_dim;                /**< 每个智能体的动作维度 */
    int hidden_size;               /**< 隐藏层大小 */
    float lr;                      /**< 学习率 */
    float gamma;                   /**< 折扣因子 */
    float tau;                     /**< 目标网络软更新系数 */
    int target_update_interval;    /**< 目标网络硬更新间隔 */
    int batch_size;                /**< 批次大小 */
    int buffer_capacity;           /**< 经验回放容量 */
    int cfc_ode_solver;            /**< CfC ODE求解器类型 */
    float cfc_time_constant;       /**< CfC时间常数 */
} MACfcVDNConfig;

/**
 * @brief CfC-COMA配置
 */
typedef struct {
    int num_agents;                /**< 智能体数量 */
    int state_dim;                 /**< 每个智能体的状态维度 */
    int action_dim;                /**< 每个智能体的动作维度 */
    int hidden_size;               /**< 隐藏层大小 */
    float actor_lr;                /**< 演员学习率 */
    float critic_lr;               /**< 评论家学习率 */
    float gamma;                   /**< 折扣因子 */
    float tau;                     /**< 软更新系数 */
    float entropy_coef;            /**< 熵系数（鼓励探索） */
    int batch_size;                /**< 批次大小 */
    int buffer_capacity;           /**< 经验回放容量 */
    int cfc_ode_solver;            /**< CfC ODE求解器类型 */
    float cfc_time_constant;       /**< CfC时间常数 */
} MACfcCOMAConfig;

/**
 * @brief CfC多智能体系统配置联合体
 */
typedef union {
    MACfcMADDPGConfig maddpg;
    MACfcQMIXConfig qmix;
    MACfcVDNConfig vdn;
    MACfcCOMAConfig coma;
} MACfcAlgorithmConfig;

/**
 * @brief CfC多智能体系统整体配置
 */
typedef struct {
    MACfcAlgorithm algorithm;           /**< 使用算法类型 */
    MACfcAlgorithmConfig algo_config;   /**< 算法配置 */
    int enable_centralized_critic;      /**< 是否启用集中式评论家 */
    int enable_communication;           /**< 是否启用智能体间通信 */
    int enable_parameter_sharing;       /**< 是否启用参数共享 */
    int global_state_dim;               /**< 全局状态维度（对于QMIX） */
} MACfcConfig;

/**
 * @brief 创建CfC多智能体系统
 * 
 * @param config 系统配置
 * @return MultiAgentSystem* 系统指针，失败返回NULL
 */
MultiAgentSystem* ma_cfc_system_create(const MACfcConfig* config);

/**
 * @brief 销毁CfC多智能体系统
 * 
 * @param system 多智能体系统
 */
void ma_cfc_system_destroy(MultiAgentSystem* system);

/**
 * @brief CfC多智能体选择动作
 * 
 * @param system 多智能体系统
 * @param states 所有智能体状态数组 [num_agents * state_dim]
 * @param actions 动作输出数组 [num_agents * action_dim]
 * @param agent_indices 参与的智能体索引（NULL表示所有智能体）
 * @param num_active 活跃智能体数量（-1表示全部）
 * @param add_noise 是否添加探索噪声
 * @return int 成功返回0，失败返回-1
 */
int ma_cfc_select_actions(MultiAgentSystem* system, const float* states,
                         float* actions, const int* agent_indices,
                         int num_active, int add_noise);

/**
 * @brief CfC多智能体训练一步
 * 
 * @param system 多智能体系统
 * @param states 联合状态 [num_agents * state_dim]
 * @param actions 联合动作 [num_agents * action_dim]
 * @param rewards 每个智能体奖励 [num_agents]
 * @param next_states 下一联合状态 [num_agents * state_dim]
 * @param dones 终止标志 [num_agents]
 * @param global_state 全局状态（QMIX需要，NULL则默认）
 * @param next_global_state 下一全局状态（QMIX需要，NULL则默认）
 * @return int 成功返回0，失败返回-1
 */
int ma_cfc_train_step(MultiAgentSystem* system, const float* states,
                     const float* actions, const float* rewards,
                     const float* next_states, const int* dones,
                     const float* global_state, const float* next_global_state);

/**
 * @brief 获取CfC多智能体系统配置默认值
 * 
 * @param config 配置输出
 * @param algorithm 算法类型
 */
void ma_cfc_default_config(MACfcConfig* config, MACfcAlgorithm algorithm);

/**
 * @brief 获取CfC多智能体算法名称
 * 
 * @param algorithm 算法类型
 * @return const char* 名称字符串
 */
const char* ma_cfc_algorithm_name(MACfcAlgorithm algorithm);

/* ============================================================================
 * ZSFZS-F024: 多智能体系统启停控制
 * ============================================================================ */

/**
 * @brief 设置多智能体系统的启用状态
 * 
 * 启用时激活智能体决策、任务调度和集体学习。
 * 禁用时停止所有智能体活动，保留现有知识和协作关系。
 * 
 * @param system 多智能体系统
 * @param enabled 1=启用, 0=禁用
 * @return int 成功返回0，失败返回-1
 */
int multi_agent_set_enabled(MultiAgentSystem* system, int enabled);

/**
 * @brief 获取多智能体系统的启用状态
 * 
 * @param system 多智能体系统
 * @return int 1=启用, 0=禁用, -1=错误
 */
int multi_agent_get_enabled(MultiAgentSystem* system);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MULTI_AGENT_H */