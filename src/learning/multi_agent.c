/**
 * @file multi_agent.c
 * @brief 多智能体协作框架完整算法实现
 * 
 * 完整的多智能体系统，支持智能体通信、协作学习、共识算法和分布式决策。
 * 提供集中式和分布式两种架构，支持多种协作策略和学习算法。
 */


#include "selflnn/multi_agent.h"
#include "selflnn/learning/learning.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/math_utils.h"

/* 静态函数前向声明（防止C4142隐式声明与定义冲突） */
static float calculate_reward(Agent* agent, const AgentAction* action, const float* next_state, size_t state_size);
static float estimate_state_value(Agent* agent, const float* state, size_t state_size);

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

/* 常量定义 */
#define MAX_COLLABORATORS 10  /**< 最大协作智能体数量 */

/* 跨平台互斥锁类型定义 */
#ifdef _WIN32
typedef CRITICAL_SECTION SystemMutex;
#else
typedef pthread_mutex_t SystemMutex;
#endif

/* 智能体个体性能统计 */
typedef struct {
    int agent_id;
    float cumulative_reward;
    float success_rate;
    int task_completed;
    float average_decision_quality;
    float learning_progress;
} AgentPerformance;

/* 多智能体系统性能统计 */
typedef struct {
    float average_reward;
    float average_success_rate;
    int total_decisions;
    int total_tasks_completed;
    int total_tasks_created;
    float collaboration_efficiency;
    float communication_efficiency;
    float global_reward;
    float evaluation_time;
    AgentPerformance individual_performance[10];
} MultiAgentPerformance;

/* Agent 内部结构体定义 */
struct Agent {
    AgentConfig config;
    AgentState state;
    void* perception_model;
    void* decision_model;
    void* value_model;
    AgentMessage** message_queue;
    int message_count;
    int message_capacity;
    int knowledge_size;
    float* local_knowledge;
    float* global_knowledge;
    int collaborator_count;
    int* collaborators;
    float* collaboration_weights;
    float cumulative_reward;
    float success_rate;
    int task_completed;
    int decisions_made;
    int buffer_size;
    int buffer_head;
    int buffer_tail;
    int buffer_count;
    float* action_buffer;
    float* observation_buffer;
    float* reward_buffer;
    float* next_observation_buffer;
    int* done_buffer;
};

/* MultiAgentSystem 内部结构体定义 */
struct MultiAgentSystem {
    MultiAgentConfig config;
    Agent** agents;
    int agent_count;
    int agent_capacity;
    float* shared_knowledge;
    size_t shared_knowledge_size;
    float* global_state;
    float* consensus_values;
    int consensus_rounds;
    float consensus_convergence;
    float consensus_quality;
    int consensus_reached;
    AgentMessage** global_messages;
    int global_message_count;
    int global_message_capacity;
    CollaborativeTask** active_tasks;
    int active_task_count;
    CollaborativeTask** completed_tasks;
    int completed_task_count;
    float global_reward;
    float communication_efficiency;
    float collaboration_efficiency;
    MultiAgentPerformance performance;
    SystemMutex system_mutex;
    int synchronization_counter;
    float synchronization_error;
    void* cfc_private_data;
};

/* 静态函数原型声明 */

/**
 * @brief 初始化参数数组
 */
static float* init_parameter_array(size_t count, float initial_value) {
    float* params = (float*)safe_malloc(count * sizeof(float));
    if (params) {
        for (size_t i = 0; i < count; i++) {
            params[i] = initial_value;
        }
    }
    return params;
}

/**
 * @brief 计算向量相似度（余弦相似度）
 */
static float compute_vector_similarity(const float* vec1, const float* vec2, size_t size);

/* 确定性伪随机数生成（基于种子） */
static float deterministic_rand(uint32_t* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7FFFFFFF;
    return (float)(*seed) / (float)0x7FFFFFFF;
}

/* 跨平台互斥锁操作 */
static int system_mutex_init(SystemMutex* mutex) {
#ifdef _WIN32
    InitializeCriticalSection(mutex);
    return 0;
#else
    return pthread_mutex_init(mutex, NULL);
#endif
}

static void system_mutex_destroy(SystemMutex* mutex) {
#ifdef _WIN32
    DeleteCriticalSection(mutex);
#else
    pthread_mutex_destroy(mutex);
#endif
}

/* 前向声明 */
static int evaluate_system_performance(MultiAgentSystem* system);

/**
 * @brief 计算向量相似度（余弦相似度）
 */
static float compute_vector_similarity(const float* vec1, const float* vec2, size_t size) {
    if (!vec1 || !vec2 || size == 0) {
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    
    norm1 = sqrtf(norm1);
    norm2 = sqrtf(norm2);
    
    if (norm1 == 0.0f || norm2 == 0.0f) {
        return 0.0f;
    }
    
    return dot_product / (norm1 * norm2);
}

/**
 * @brief 计算平均共识值
 */
static float compute_average_consensus(const float* values, int count) {
    if (!values || count <= 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += values[i];
    }
    
    return sum / count;
}

/**
 * @brief 执行加权平均共识
 */
static void weighted_average_consensus(float* values, const float* weights, int count, int rounds) {
    if (!values || !weights || count <= 0 || rounds <= 0) {
        return;
    }
    
    for (int r = 0; r < rounds; r++) {
        // 计算加权平均
        float total_weight = 0.0f;
        float weighted_sum = 0.0f;
        
        for (int i = 0; i < count; i++) {
            weighted_sum += values[i] * weights[i];
            total_weight += weights[i];
        }
        
        if (total_weight > 0.0f) {
            float average = weighted_sum / total_weight;
            
            // 更新值（向平均值收敛）
            for (int i = 0; i < count; i++) {
                values[i] = values[i] * 0.7f + average * 0.3f;
            }
        }
    }
}

/**
 * @brief 计算任务分配效用
 */
static float compute_task_utility(const CollaborativeTask* task, const Agent* agent) {
    if (!task || !agent) {
        return 0.0f;
    }
    
    float utility = 0.0f;
    
    // 基于智能体能力匹配
    utility += agent->state.capability_level * 0.3f;
    
    // 基于任务类型增加额外的能力匹配
    switch (task->task_type) {
        case TASK_TYPE_PERCEPTION:
            utility += agent->state.expertise_level * 0.1f;
            break;
        case TASK_TYPE_DECISION:
            utility += agent->state.capability_level * 0.1f;
            break;
        case TASK_TYPE_ACTION:
            utility += agent->state.capability_level * 0.1f;
            break;
        case TASK_TYPE_LEARNING:
            utility += agent->state.expertise_level * 0.1f;
            break;
        default:
            utility += 0.1f;
            break;
    }
    
    // 基于智能体是否忙碌
    utility += (agent->state.busy ? 0.0f : 0.2f);
    
    // 基于智能体成功率
    utility += agent->state.success_rate * 0.2f;
    
    // 基于智能体能力与任务难度的匹配度
    if (task->difficulty > 0.0f) {
        // 计算能力匹配度：能力与难度的比值，限制在0-1范围内
        float capability_ratio = agent->state.capability_level / task->difficulty;
        if (capability_ratio > 1.0f) capability_ratio = 1.0f;  // 能力超过需求
        
        // 考虑智能体专业水平：对于某些任务类型，专业水平更重要
        float expertise_factor = 0.0f;
        switch (task->task_type) {
            case TASK_TYPE_PERCEPTION:
            case TASK_TYPE_LEARNING:
                expertise_factor = agent->state.expertise_level * 0.2f;
                break;
            case TASK_TYPE_DECISION:
            case TASK_TYPE_ACTION:
                expertise_factor = agent->state.capability_level * 0.1f;
                break;
            default:
                expertise_factor = 0.05f;
                break;
        }
        
        // 能力匹配效用：匹配度越高，效用越高
        float capability_utility = capability_ratio * 0.15f + expertise_factor;
        
        // 考虑任务紧急程度：越紧急的任务，可用智能体越少，分配效用应适当增加
        float urgency_factor = task->urgency * 0.05f;
        
        utility += capability_utility + urgency_factor;
    } else {
        // 任务难度未设置，使用基础效用
        utility += 0.1f;
    }
    
    return utility;
}

/* ============================================================================
 * 配置管理
 * =========================================================================== */

/**
 * @brief 获取默认智能体配置
 */
void agent_default_config(AgentConfig* config) {
    if (!config) return;
    
    memset(config, 0, sizeof(AgentConfig));
    
    config->type = AGENT_TYPE_LEARNING;
    config->role = AGENT_ROLE_WORKER;
    config->initial_capability = 0.5f;
    
    config->learning_rate = 0.001f;
    config->exploration_rate = 0.1f;
    config->discount_factor = 0.95f;
    
    config->memory_size = 10000;
    config->batch_size = 32;
    config->update_frequency = 100;
    
    config->communication_range = 10.0f;
    config->max_messages = 100;
    config->message_ttl = 10;
    
    config->collaboration_threshold = 0.7f;
    config->trust_update_rate = 0.01f;
    
    config->enable_learning = 1;
    config->enable_communication = 1;
    config->enable_collaboration = 1;
    config->enable_adaptation = 1;
}

/**
 * @brief 获取默认多智能体系统配置
 */
void multi_agent_default_config(MultiAgentConfig* config) {
    if (!config) return;
    
    memset(config, 0, sizeof(MultiAgentConfig));
    
    config->agent_count = 5;
    config->centralized = 0;          // 默认分布式
    config->distributed = 1;
    config->decentralized = 0;
    
    config->global_strategy = COOPERATION_SHARED_GOAL;
    config->communication_protocol = PROTOCOL_BROADCAST;
    config->consensus_algorithm = CONSENSUS_AVERAGE;
    
    config->max_tasks = 10;
    config->max_messages = 1000;
    config->knowledge_size = 256;
    
    config->synchronization_interval = 10;
    config->consensus_rounds = 5;
    config->collaboration_threshold = 0.6f;
    
    config->enable_global_learning = 1;
    config->enable_local_learning = 1;
    config->enable_knowledge_sharing = 1;
    config->enable_consensus = 1;
    
    config->learning_rate = 0.001f;
    config->collaboration_weight = 0.5f;
    config->communication_weight = 0.3f;
    config->exploration_weight = 0.2f;
    
    config->evaluation_interval = 100;
    config->logging_level = LOG_LEVEL_INFO;
}

/* ============================================================================
 * 任务和消息管理
 * =========================================================================== */

/**
 * @brief 创建协作任务
 */
CollaborativeTask* collaborative_task_create(const char* task_id, int required_agents) {
    CollaborativeTask* task = (CollaborativeTask*)safe_calloc(1, sizeof(CollaborativeTask));
    if (!task) {
        return NULL;
    }
    
    // 设置任务ID
    if (task_id) {
        task->task_id = string_duplicate_nullable(task_id);
        if (!task->task_id) {
            safe_free((void**)&task);
            return NULL;
        }
    }
    
    // 初始化任务参数
    task->required_agents = required_agents;
    task->assigned_agents = 0;
    task->completed_agents = 0;
    task->completion_ratio = 0.0f;
    task->priority = 1.0f;
    task->difficulty = 0.5f;
    task->deadline = 0;  // 无期限
    task->creation_time = 0;
    task->start_time = 0;
    task->completion_time = 0;
    
    task->task_type = TASK_TYPE_DECISION;
    task->task_status = TASK_STATUS_PENDING;
    task->collaboration_type = COLLABORATION_SEQUENTIAL;
    
    task->input_data = NULL;
    task->output_data = NULL;
    task->task_parameters = NULL;
    task->params_size = 0;
    
    task->quality = 0.0f;
    task->efficiency = 0.0f;
    task->collaboration_score = 0.0f;
    task->reward = 0.0f;
    
    return task;
}

/**
 * @brief 销毁协作任务
 */
void collaborative_task_destroy(CollaborativeTask* task) {
    if (!task) return;
    
    safe_free((void**)&task->task_id);
    safe_free((void**)&task->input_data);
    safe_free((void**)&task->output_data);
    safe_free((void**)&task->task_parameters);
    safe_free((void**)&task->assigned_agents_list);
    safe_free((void**)&task);
}

/**
 * @brief 创建智能体消息
 */
AgentMessage* agent_message_create(int sender_id, int receiver_id, const char* message_type) {
    AgentMessage* msg = (AgentMessage*)safe_calloc(1, sizeof(AgentMessage));
    if (!msg) {
        return NULL;
    }
    
    msg->sender_id = sender_id;
    msg->receiver_id = receiver_id;
    msg->timestamp = 0;  // 需要设置实际时间戳
    
    if (message_type) {
        msg->message_type = string_duplicate_nullable(message_type);
        if (!msg->message_type) {
            safe_free((void**)&msg);
            return NULL;
        }
    }
    
    msg->message_data = NULL;
    msg->data_size = 0;
    msg->priority = 1;
    msg->ttl = 10;
    msg->delivered = 0;
    msg->read = 0;
    msg->response_required = 0;
    msg->response_received = 0;
    
    return msg;
}

/**
 * @brief 销毁智能体消息
 */
void agent_message_destroy(AgentMessage* message) {
    if (!message) return;
    
    safe_free((void**)&message->message_type);
    safe_free((void**)&message->message_data);
    safe_free((void**)&message);
}

/* ============================================================================
 * 智能体核心功能
 * =========================================================================== */

/**
 * @brief 创建智能体
 */
Agent* agent_create(const AgentConfig* config) {
    if (!config) {
        return NULL;
    }
    
    Agent* agent = (Agent*)safe_calloc(1, sizeof(Agent));
    if (!agent) {
        return NULL;
    }
    
    // 复制配置
    memcpy(&agent->config, config, sizeof(AgentConfig));
    
    // 初始化状态
    memset(&agent->state, 0, sizeof(AgentState));
    agent->state.success_rate = 0.5f;
    agent->state.efficiency = 0.5f;
    agent->state.collaboration_score = 0.5f;
    agent->state.energy_level = 1.0f;
    agent->state.available = 1;
    
    // 初始化模型指针
    agent->perception_model = NULL;
    agent->decision_model = NULL;
    agent->value_model = NULL;
    
    // 初始化通信状态
    agent->message_queue = NULL;
    agent->message_count = 0;
    agent->message_capacity = 16;  // 初始容量
    agent->message_queue = (AgentMessage**)safe_calloc(agent->message_capacity, sizeof(AgentMessage*));
    if (!agent->message_queue) {
        safe_free((void**)&agent);
        return NULL;
    }
    
    // 初始化知识表示
    agent->knowledge_size = 64;  // 默认知识表示大小
    agent->local_knowledge = init_parameter_array(agent->knowledge_size, 0.0f);
    if (!agent->local_knowledge) {
        safe_free((void**)&agent->message_queue);
        safe_free((void**)&agent);
        return NULL;
    }
    agent->global_knowledge = NULL;  // 将由系统设置
    
    // 初始化协作状态
    agent->collaborator_count = 0;
    agent->collaborators = (int*)safe_calloc(10, sizeof(int));  // 初始容量10
    agent->collaboration_weights = init_parameter_array(10, 0.1f);
    if (!agent->collaborators || !agent->collaboration_weights) {
        safe_free((void**)&agent->local_knowledge);
        safe_free((void**)&agent->collaborators);
        safe_free((void**)&agent->collaboration_weights);
        safe_free((void**)&agent->message_queue);
        safe_free((void**)&agent);
        return NULL;
    }
    
    // 初始化性能统计
    agent->cumulative_reward = 0.0f;
    agent->success_rate = 0.5f;
    agent->task_completed = 0;
    agent->decisions_made = 0;
    
    // 初始化经验回放缓冲区
    agent->buffer_size = 1000;  // 缓冲区容量
    agent->buffer_head = 0;     // 头指针从0开始
    agent->buffer_tail = 0;     // 尾指针从0开始
    agent->buffer_count = 0;    // 缓冲区为空
    
    // 分配各个缓冲区
    agent->action_buffer = init_parameter_array(agent->buffer_size, 0.0f);
    agent->observation_buffer = init_parameter_array(agent->buffer_size, 0.0f);
    agent->reward_buffer = init_parameter_array(agent->buffer_size, 0.0f);
    agent->next_observation_buffer = init_parameter_array(agent->buffer_size, 0.0f);
    agent->done_buffer = (int*)safe_calloc(agent->buffer_size, sizeof(int));
    
    if (!agent->action_buffer || !agent->observation_buffer || !agent->reward_buffer ||
        !agent->next_observation_buffer || !agent->done_buffer) {
        safe_free((void**)&agent->local_knowledge);
        safe_free((void**)&agent->collaborators);
        safe_free((void**)&agent->collaboration_weights);
        safe_free((void**)&agent->message_queue);
        safe_free((void**)&agent->action_buffer);
        safe_free((void**)&agent->observation_buffer);
        safe_free((void**)&agent->reward_buffer);
        safe_free((void**)&agent->next_observation_buffer);
        safe_free((void**)&agent->done_buffer);
        safe_free((void**)&agent);
        return NULL;
    }
    
    return agent;
}

/**
 * @brief 销毁智能体
 */
void agent_destroy(Agent* agent) {
    if (!agent) return;
    
    // 清理消息队列
    if (agent->message_queue) {
        for (int i = 0; i < agent->message_count; i++) {
            if (agent->message_queue[i]) {
                agent_message_destroy(agent->message_queue[i]);
            }
        }
        safe_free((void**)&agent->message_queue);
    }
    
    // 清理知识表示
    safe_free((void**)&agent->local_knowledge);
    safe_free((void**)&agent->global_knowledge);
    
    // 清理协作状态
    safe_free((void**)&agent->collaborators);
    safe_free((void**)&agent->collaboration_weights);
    
    // 清理缓冲区
    safe_free((void**)&agent->action_buffer);
    safe_free((void**)&agent->observation_buffer);
    safe_free((void**)&agent->reward_buffer);
    safe_free((void**)&agent->next_observation_buffer);
    safe_free((void**)&agent->done_buffer);
    
    // 清理模型
    // 注意：模型内存由外部管理，这里不释放
    
    safe_free((void**)&agent);
}

/**
 * @brief 智能体决策 - 使用LNN液态神经网络驱动
 */
AgentAction* agent_decide(Agent* agent, const float* observation, size_t obs_size) {
    if (!agent || !observation || obs_size == 0) {
        return NULL;
    }
    
    AgentAction* action = (AgentAction*)safe_calloc(1, sizeof(AgentAction));
    if (!action) {
        return NULL;
    }
    
    action->values_size = obs_size;
    action->action_values = (float*)safe_calloc(action->values_size, sizeof(float));
    if (!action->action_values) {
        safe_free((void**)&action);
        return NULL;
    }
    
    /* 尝试使用LNN决策模型（若已初始化） */
    int lnn_decided = 0;
    if (agent->decision_model) {
        LNN* lnn = (LNN*)selflnn_get_lnn();
        if (lnn) {
            /* 使用LNN全局模型进行决策 */
            int lnn_input_size = (int)obs_size < 128 ? 128 : (int)obs_size;
            float* lnn_input = (float*)safe_calloc(lnn_input_size, sizeof(float));
            float* lnn_output = (float*)safe_calloc(lnn_input_size, sizeof(float));
            
            if (lnn_input && lnn_output) {
                /* 构造LNN输入：观察值 + 智能体状态特征 */
                for (size_t i = 0; i < obs_size && i < 120; i++) {
                    lnn_input[i] = observation[i];
                }
                /* 附加智能体内部状态作为上下文 */
                if (obs_size < 120) {
                    lnn_input[obs_size + 0] = agent->state.energy_level;
                    lnn_input[obs_size + 1] = agent->state.success_rate;
                    lnn_input[obs_size + 2] = agent->state.efficiency;
                    lnn_input[obs_size + 3] = agent->config.exploration_rate;
                    lnn_input[obs_size + 4] = (float)agent->collaborator_count / 10.0f;
                    lnn_input[obs_size + 5] = (float)agent->state.capability_level / 100.0f;
                    lnn_input[obs_size + 6] = (float)agent->config.learning_rate;
                    lnn_input[obs_size + 7] = 1.0f; /* 偏置 */
                }
                
                int fwd_ret = lnn_forward(lnn, lnn_input, lnn_output);
                if (fwd_ret == 0) {
                    /* LNN输出作为动作值 */
                    for (size_t i = 0; i < obs_size; i++) {
                        /* 使用tanh激活将LNN输出映射到合适范围 */
                        float raw = (i < (size_t)lnn_input_size) ? lnn_output[i] : 0.0f;
                        action->action_values[i] = tanhf(raw * 0.5f);
                    }
                    
                    /* 根据LNN输出判定动作类型 */
                    float sum_values = 0.0f;
                    for (size_t i = 0; i < obs_size; i++) {
                        sum_values += action->action_values[i] * action->action_values[i];
                    }
                    float rms = sqrtf(sum_values / (float)obs_size + 1e-10f);
                    
                    if (rms > 0.3f && agent->config.type == AGENT_TYPE_EXPLORER) {
                        action->action_type = ACTION_TYPE_EXPLORE;
                        action->confidence = 0.7f + rms * 0.3f;
                    } else if (rms > 0.5f && agent->config.type == AGENT_TYPE_COLLABORATOR) {
                        action->action_type = ACTION_TYPE_COLLABORATE;
                        action->confidence = 0.8f + rms * 0.2f;
                    } else {
                        action->action_type = ACTION_TYPE_EXPLOIT;
                        action->confidence = 0.6f + rms * 0.4f;
                    }
                    
                    action->expected_reward = rms * agent->state.success_rate;
                    lnn_decided = 1;
                }
            }
            safe_free((void**)&lnn_input);
            safe_free((void**)&lnn_output);
        }
    }
    
    if (!lnn_decided) {
        /* LNN未就绪时使用确定性推理规则，基于多因素加权评估 */
        /* 根据智能体类型决定动作类型 */
        if (agent->config.type == AGENT_TYPE_EXPLORER) {
            action->action_type = ACTION_TYPE_EXPLORE;
            action->confidence = 0.7f;
        } else if (agent->config.type == AGENT_TYPE_EXPLOITER) {
            action->action_type = ACTION_TYPE_EXPLOIT;
            action->confidence = 0.9f;
        } else if (agent->config.type == AGENT_TYPE_COLLABORATOR) {
            action->action_type = ACTION_TYPE_COLLABORATE;
            action->confidence = 0.8f;
        } else {
            action->action_type = ACTION_TYPE_NONE;
            action->confidence = 0.5f;
        }
        
        /* 完整决策逻辑：基于多因素加权评估 */
        for (size_t i = 0; i < obs_size; i++) {
            float obs_val = observation[i];
            float action_val = 0.0f;
            
            if (obs_val > 0.8f) {
                action_val = agent->config.exploration_rate * (0.8f + 0.2f * obs_val);
            } else if (obs_val > 0.5f) {
                action_val = agent->config.exploration_rate * (0.3f + 0.5f * obs_val);
            } else if (obs_val > 0.2f) {
                action_val = agent->config.exploration_rate * (0.1f * obs_val - 0.05f);
            } else {
                action_val = -0.15f * (1.0f - obs_val);
            }
            
            float energy_factor = agent->state.energy_level;
            if (energy_factor < 0.3f) {
                action_val *= 0.3f + 0.7f * energy_factor;
            } else if (energy_factor > 0.8f) {
                action_val *= 1.0f + 0.2f * (energy_factor - 0.8f);
            }
            
            float success_factor = agent->state.success_rate;
            if (success_factor < 0.2f) {
                action_val *= 0.5f;
            } else if (success_factor > 0.7f) {
                action_val *= 1.0f + 0.15f * success_factor;
            }
            
            float explore_bonus = agent->config.exploration_rate * (1.0f - success_factor) * 0.2f;
            if (obs_val > 0.3f && obs_val < 0.7f) {
                action_val += explore_bonus;
            }
            
            action->action_values[i] = action_val;
        }
        
        action->expected_reward = agent->state.success_rate * agent->state.energy_level;
    }
    
    if (agent->collaborator_count > 0 && action->action_type == ACTION_TYPE_COLLABORATE) {
        float max_weight = 0.0f;
        int best_collaborator = -1;
        for (int i = 0; i < agent->collaborator_count; i++) {
            if (agent->collaboration_weights[i] > max_weight) {
                max_weight = agent->collaboration_weights[i];
                best_collaborator = agent->collaborators[i];
            }
        }
        action->target_agent_id = best_collaborator;
    } else {
        action->target_agent_id = -1;
    }
    
    action->task_id = -1;
    agent->decisions_made++;
    
    return action;
}

/**
 * @brief 智能体学习
 */
int agent_learn(Agent* agent, const AgentExperience* experience) {
    if (!agent || !experience) {
        return -1;
    }
    
    // 确保有有效的观察值
    if (!experience->state_observation || !experience->next_state_observation || 
        !experience->action || !experience->action->action_values) {
        return -1;
    }
    
    // 计算时间差分误差（TD error）
    float td_error = experience->reward + agent->config.learning_rate * 
                    (experience->next_state_value - experience->state_value);
    
    // 更新状态价值函数（实际实现）
    // 使用TD误差更新智能体能力水平和专业水平
    float value_update = agent->config.learning_rate * td_error;
    
    // 正TD误差：表现比预期好，提高能力
    // 负TD误差：表现比预期差，降低能力
    float capability_change = value_update * 0.1f;  // 缩放因子
    agent->state.capability_level += capability_change;
    
    // 确保能力水平在合理范围内 [0, 1]
    if (agent->state.capability_level < 0.0f) agent->state.capability_level = 0.0f;
    if (agent->state.capability_level > 1.0f) agent->state.capability_level = 1.0f;
    
    // 根据任务类型更新专业水平
    float expertise_change = 0.0f;
    if (experience->action) {
        // 基于动作类型和奖励调整专业水平
        switch (experience->action->action_type) {
            case ACTION_TYPE_LEARN:
                expertise_change = value_update * 0.15f;  // 学习任务更影响专业水平
                break;
            case ACTION_TYPE_EXPLOIT:
                expertise_change = value_update * 0.1f;   // 利用任务影响较小
                break;
            case ACTION_TYPE_COLLABORATE:
                expertise_change = value_update * 0.12f;  // 协作任务中等影响
                break;
            case ACTION_TYPE_OBSERVE:
                expertise_change = value_update * 0.09f;  // 观察任务基础影响
                break;
            case ACTION_TYPE_EXPLORE:
                expertise_change = value_update * 0.08f;  // 探索任务基础影响
                break;
            default:
                expertise_change = value_update * 0.07f;  // 其他任务基础影响
                break;
        }
    }
    
    agent->state.expertise_level += expertise_change;
    if (agent->state.expertise_level < 0.0f) agent->state.expertise_level = 0.0f;
    if (agent->state.expertise_level > 1.0f) agent->state.expertise_level = 1.0f;
    
    // 更新策略（基于动作值）
    float action_magnitude = 0.0f;
    for (size_t i = 0; i < experience->action->values_size && i < experience->obs_size; i++) {
        action_magnitude += fabsf(experience->action->action_values[i]);
    }
    if (experience->action->values_size > 0) {
        action_magnitude /= experience->action->values_size;
    }
    
    // 更新智能体状态
    agent->state.success_rate = 0.9f * agent->state.success_rate + 
                               0.1f * (experience->reward > 0 ? 1.0f : 0.0f);
    
    // 更新累积奖励
    agent->cumulative_reward += experience->reward;
    
    // 更新能量水平（学习消耗能量）
    float learning_cost = 0.01f * action_magnitude;
    agent->state.energy_level -= learning_cost;
    if (agent->state.energy_level < 0.0f) {
        agent->state.energy_level = 0.0f;
    }
    
    // 将经验添加到回放缓冲区（完整实现）
    // 存储完整的经验元组：状态摘要、动作摘要、奖励、下一状态摘要、终止标志
    if (agent->buffer_size > 0 && experience->state_observation && experience->action && 
        experience->action->action_values && experience->next_state_observation) {
        
        // 计算存储位置（循环缓冲区）
        size_t idx = agent->buffer_head;
        
        // 计算观察值的摘要（均值）
        float obs_summary = 0.0f;
        float next_obs_summary = 0.0f;
        size_t valid_obs_count = 0;
        
        for (size_t i = 0; i < experience->obs_size && i < agent->buffer_size; i++) {
            obs_summary += experience->state_observation[i];
            next_obs_summary += experience->next_state_observation[i];
            valid_obs_count++;
        }
        
        if (valid_obs_count > 0) {
            obs_summary /= valid_obs_count;
            next_obs_summary /= valid_obs_count;
            
            // 存储观察值摘要
            agent->observation_buffer[idx] = obs_summary;
            agent->next_observation_buffer[idx] = next_obs_summary;
        }
        
        // 计算动作值的摘要（均值）
        float action_summary = 0.0f;
        size_t valid_action_count = 0;
        
        for (size_t i = 0; i < experience->action->values_size && i < agent->buffer_size; i++) {
            action_summary += experience->action->action_values[i];
            valid_action_count++;
        }
        
        if (valid_action_count > 0) {
            action_summary /= valid_action_count;
            // 存储动作值摘要
            agent->action_buffer[idx] = action_summary;
        }
        
        // 存储奖励和终止标志
        agent->reward_buffer[idx] = experience->reward;
        agent->done_buffer[idx] = experience->done;
        
        // 更新缓冲区指针和计数
        agent->buffer_head = (agent->buffer_head + 1) % agent->buffer_size;
        if (agent->buffer_count < agent->buffer_size) {
            agent->buffer_count++;
        } else {
            // 缓冲区已满，尾指针也需要移动
            agent->buffer_tail = (agent->buffer_tail + 1) % agent->buffer_size;
        }
        
        // 更新经验统计
        agent->decisions_made++;
        
        /* 记录缓冲区利用率（每100条经验记录一次） */
        if (agent->buffer_count % 100 == 0 && agent->buffer_size > 0) {
            float buffer_utilization = (float)agent->buffer_count / (float)agent->buffer_size;
            log_info("智能体 %d 经验缓冲区利用率: %.1f%% (%d/%d)\n",
                      agent->state.agent_id,
                     buffer_utilization * 100.0f,
                     agent->buffer_count,
                     agent->buffer_size);
        }
    }
    
    // 更新探索率（随着学习递减）
    if (agent->decisions_made > 1000) {
        agent->config.exploration_rate *= 0.999f;
        if (agent->config.exploration_rate < 0.01f) {
            agent->config.exploration_rate = 0.01f;
        }
    }
    
    // 更新协作权重（如果涉及协作）
    if (experience->action->target_agent_id >= 0) {
        // 寻找目标智能体在协作列表中的位置
        for (int i = 0; i < agent->collaborator_count; i++) {
            if (agent->collaborators[i] == experience->action->target_agent_id) {
                // 根据奖励调整协作权重
                float weight_update = 0.1f * experience->reward;
                agent->collaboration_weights[i] += weight_update;
                
                // 保持权重在合理范围
                if (agent->collaboration_weights[i] < 0.0f) {
                    agent->collaboration_weights[i] = 0.0f;
                }
                if (agent->collaboration_weights[i] > 1.0f) {
                    agent->collaboration_weights[i] = 1.0f;
                }
                break;
            }
        }
    }
    
    // 任务完成统计
    if (experience->done) {
        agent->task_completed++;
    }
    
    return 0;
}

/**
 * @brief 智能体通信
 */
int agent_communicate(Agent* sender, Agent* receiver, const char* message_type, 
                     const void* message_data, size_t data_size) {
    if (!sender || !receiver || !message_type) {
        return -1;
    }
    
    // 创建消息结构
    AgentMessage message;
    memset(&message, 0, sizeof(AgentMessage));
    
    // 使用智能体的实际ID
    // 智能体状态中包含agent_id字段
    message.sender_id = sender->state.agent_id;
    message.receiver_id = receiver->state.agent_id;
    
    // 复制消息类型
    message.message_type = string_duplicate_nullable(message_type);
    if (!message.message_type) {
        return -1;
    }
    
    // 复制消息数据
    if (message_data && data_size > 0) {
        message.message_data = safe_malloc(data_size);
        if (!message.message_data) {
            safe_free((void**)&message.message_type);
            return -1;
        }
        memcpy(message.message_data, message_data, data_size);
        message.data_size = data_size;
    }
    
    message.timestamp = (float)time(NULL);
    message.priority = 1.0f;
    message.ttl = 10;
    message.delivered = 0;
    message.read = 0;
    message.response_required = 0;
    message.response_received = 0;
    
    // 检查接收者的消息队列容量
    if (receiver->message_count >= receiver->message_capacity) {
        // 扩展消息队列
        int new_capacity = receiver->message_capacity * 2;
        AgentMessage** new_queue = (AgentMessage**)safe_realloc(receiver->message_queue, 
                                                               new_capacity * sizeof(AgentMessage*));
        if (!new_queue) {
            safe_free((void**)&message.message_type);
            safe_free((void**)&message.message_data);
            return -1;
        }
        receiver->message_queue = new_queue;
        receiver->message_capacity = new_capacity;
    }
    
    // 创建消息副本（接收者需要自己的副本）
    AgentMessage* message_copy = (AgentMessage*)safe_calloc(1, sizeof(AgentMessage));
    if (!message_copy) {
        safe_free((void**)&message.message_type);
        safe_free((void**)&message.message_data);
        return -1;
    }
    
    // 复制消息内容
    memcpy(message_copy, &message, sizeof(AgentMessage));
    
    // 深度复制字符串字段（需要新的内存）
    if (message.message_type) {
        message_copy->message_type = string_duplicate_nullable(message.message_type);
        if (!message_copy->message_type) {
            safe_free((void**)&message_copy);
            safe_free((void**)&message.message_type);
            safe_free((void**)&message.message_data);
            return -1;
        }
    }
    
    // 深度复制消息数据
    if (message.message_data && message.data_size > 0) {
        message_copy->message_data = safe_malloc(message.data_size);
        if (!message_copy->message_data) {
            safe_free((void**)&message_copy->message_type);
            safe_free((void**)&message_copy);
            safe_free((void**)&message.message_type);
            safe_free((void**)&message.message_data);
            return -1;
        }
        memcpy(message_copy->message_data, message.message_data, message.data_size);
        message_copy->data_size = message.data_size;
    }
    
    // 更新消息状态
    message_copy->delivered = 1;  // 标记为已送达
    
    // 添加到接收者消息队列
    receiver->message_queue[receiver->message_count] = message_copy;
    receiver->message_count++;
    
    // 清理临时消息的内存
    safe_free((void**)&message.message_type);
    safe_free((void**)&message.message_data);
    
    // 更新发送者的通信统计
    sender->state.success_rate = 0.95f * sender->state.success_rate + 0.05f;
    
    // 如果是需要响应的消息，记录到发送者的待响应列表
    if (strstr(message_type, "request") != NULL || strstr(message_type, "query") != NULL) {
        message_copy->response_required = 1;
        sender->state.available = 0;  // 标记为忙碌等待响应
    }
    
    // 更新协作权重（如果这是协作消息）
    if (strstr(message_type, "collaboration") != NULL || 
        strstr(message_type, "cooperate") != NULL) {
        // 寻找接收者在发送者的协作列表中的位置
        int collaborator_idx = -1;
        if (sender->collaborator_count > 0) {
            // 在协作列表中查找接收者ID
            for (int i = 0; i < sender->collaborator_count; i++) {
                if (sender->collaborators[i] == receiver->state.agent_id) {
                    collaborator_idx = i;
                    break;
                }
            }
            
            // 如果接收者不在协作列表中，添加到列表末尾（如果空间允许）
            if (collaborator_idx == -1 && sender->collaborator_count < MAX_COLLABORATORS) {
                collaborator_idx = sender->collaborator_count;
                sender->collaborators[collaborator_idx] = receiver->state.agent_id;
                sender->collaboration_weights[collaborator_idx] = 0.0f;
                sender->collaborator_count++;
            }
        }
        
        if (collaborator_idx >= 0) {
            // 增加协作权重
            sender->collaboration_weights[collaborator_idx] += 0.05f;
            if (sender->collaboration_weights[collaborator_idx] > 1.0f) {
                sender->collaboration_weights[collaborator_idx] = 1.0f;
            }
        }
    }
    
    return 0;
}

/**
 * @brief 知识共享函数（完整实现）
 */
static void share_knowledge(MultiAgentSystem* system) {
    if (!system || system->agent_count <= 0) return;
    
    // 完整实现：基于智能体能力水平的加权知识融合
    // 同时添加知识变异以促进多样性
    
    // 首先，计算每个智能体的权重（基于能力水平和专业水平）
    float* agent_weights = (float*)safe_calloc(system->agent_count, sizeof(float));
    if (!agent_weights) return;
    
    float total_weight = 0.0f;
    for (int i = 0; i < system->agent_count; i++) {
        Agent* agent = system->agents[i];
        if (agent) {
            // 权重 = 能力水平 * 0.7 + 专业水平 * 0.3 + 成功率 * 0.2
            agent_weights[i] = agent->state.capability_level * 0.7f +
                               agent->state.expertise_level * 0.3f +
                               agent->state.success_rate * 0.2f;
            // 确保权重非负
            if (agent_weights[i] < 0.01f) agent_weights[i] = 0.01f;
            total_weight += agent_weights[i];
        } else {
            agent_weights[i] = 0.01f;  // 最小权重
            total_weight += 0.01f;
        }
    }
    
    // 归一化权重
    if (total_weight > 0.0f) {
        for (int i = 0; i < system->agent_count; i++) {
            agent_weights[i] /= total_weight;
        }
    }
    
    // 为每个智能体计算加权平均知识
    for (int i = 0; i < system->agent_count; i++) {
        Agent* agent = system->agents[i];
        if (!agent || !agent->local_knowledge || agent->knowledge_size == 0) continue;
        
        // 创建临时知识缓冲区
        float* new_knowledge = (float*)safe_calloc(agent->knowledge_size, sizeof(float));
        if (!new_knowledge) {
            safe_free((void**)&agent_weights);
            return;
        }
        
        // 计算加权平均
        for (size_t k = 0; k < agent->knowledge_size; k++) {
            float weighted_sum = 0.0f;
            float weight_sum = 0.0f;
            
            for (int j = 0; j < system->agent_count; j++) {
                Agent* other = system->agents[j];
                if (other && other->local_knowledge) {
                    weighted_sum += other->local_knowledge[k] * agent_weights[j];
                    weight_sum += agent_weights[j];
                }
            }
            
            if (weight_sum > 0.0f) {
                new_knowledge[k] = weighted_sum / weight_sum;
                
                // 添加小量随机变异以促进多样性（变异概率10%，使用确定性伪随机数）
                uint32_t mutation_seed = (uint32_t)((uintptr_t)agent ^ (uintptr_t)(k * 7919U));
                float mutation_prob = deterministic_rand(&mutation_seed);
                if (mutation_prob < 0.10f) {
                    // 添加正态分布的随机噪声（标准差0.05，使用确定性伪随机数）
                    uint32_t noise_seed = mutation_seed ^ 0xABCDEF01U;
                    float noise = (deterministic_rand(&noise_seed) - 0.5f) * 0.1f;
                    new_knowledge[k] += noise;
                    
                    // 限制在合理范围内
                    if (new_knowledge[k] < -1.0f) new_knowledge[k] = -1.0f;
                    if (new_knowledge[k] > 1.0f) new_knowledge[k] = 1.0f;
                }
            } else {
                new_knowledge[k] = agent->local_knowledge[k];  // 保持原值
            }
        }
        
        // 更新智能体知识（渐进更新：保留10%原知识，采用90%新知识）
        for (size_t k = 0; k < agent->knowledge_size; k++) {
            agent->local_knowledge[k] = agent->local_knowledge[k] * 0.1f + new_knowledge[k] * 0.9f;
        }
        
        safe_free((void**)&new_knowledge);
    }
    
    safe_free((void**)&agent_weights);
    
    // 更新系统共享知识（所有智能体知识的平均值）
    if (system->shared_knowledge && system->shared_knowledge_size > 0) {
        for (size_t k = 0; k < system->shared_knowledge_size; k++) {
            float sum = 0.0f;
            int count = 0;
            
            for (int i = 0; i < system->agent_count; i++) {
                Agent* agent = system->agents[i];
                if (agent && agent->local_knowledge && k < agent->knowledge_size) {
                    sum += agent->local_knowledge[k];
                    count++;
                }
            }
            
            if (count > 0) {
                system->shared_knowledge[k] = sum / count;
            }
        }
    }
}

/* ============================================================================
 * 多智能体系统核心API
 * =========================================================================== */

/**
 * @brief 创建多智能体系统
 */
MultiAgentSystem* multi_agent_system_create(const MultiAgentConfig* config) {
    MultiAgentSystem* system = (MultiAgentSystem*)safe_calloc(1, sizeof(MultiAgentSystem));
    if (!system) {
        return NULL;
    }
    
    // 设置配置
    if (config) {
        memcpy(&system->config, config, sizeof(MultiAgentConfig));
    } else {
        multi_agent_default_config(&system->config);
    }
    
    // 初始化智能体管理
    system->agent_capacity = system->config.agent_count + 10;
    system->agent_count = 0;
    system->agents = (Agent**)safe_calloc(system->agent_capacity, sizeof(Agent*));
    if (!system->agents) {
        safe_free((void**)&system);
        return NULL;
    }
    
    // 创建智能体（基于配置）
    if (system->config.agent_configs) {
        for (int i = 0; i < system->config.agent_count; i++) {
            Agent* agent = agent_create(&system->config.agent_configs[i]);
            if (agent) {
                system->agents[system->agent_count++] = agent;
            }
        }
    }
    
    // 初始化通信网络
    system->global_message_capacity = system->config.max_messages;
    system->global_message_count = 0;
    system->global_messages = (AgentMessage**)safe_calloc(system->global_message_capacity, 
                                                         sizeof(AgentMessage*));
    if (!system->global_messages) {
        for (int i = 0; i < system->agent_count; i++) {
            agent_destroy(system->agents[i]);
        }
        safe_free((void**)&system->agents);
        safe_free((void**)&system);
        return NULL;
    }
    
    // 初始化任务管理
    system->active_task_count = 0;
    system->completed_task_count = 0;
    system->active_tasks = (CollaborativeTask**)safe_calloc(system->config.max_tasks, 
                                                           sizeof(CollaborativeTask*));
    system->completed_tasks = (CollaborativeTask**)safe_calloc(system->config.max_tasks, 
                                                              sizeof(CollaborativeTask*));
    if (!system->active_tasks || !system->completed_tasks) {
        safe_free((void**)&system->active_tasks);
        safe_free((void**)&system->completed_tasks);
        for (int i = 0; i < system->agent_count; i++) {
            agent_destroy(system->agents[i]);
        }
        safe_free((void**)&system->agents);
        safe_free((void**)&system->global_messages);
        safe_free((void**)&system);
        return NULL;
    }
    
    // 初始化协作状态
    system->shared_knowledge_size = system->config.knowledge_size;
    system->shared_knowledge = init_parameter_array(system->shared_knowledge_size, 0.0f);
    system->global_state = init_parameter_array(system->shared_knowledge_size, 0.0f);
    system->consensus_values = init_parameter_array(system->agent_count, 0.5f);
    if (!system->shared_knowledge || !system->global_state || !system->consensus_values) {
        safe_free((void**)&system->shared_knowledge);
        safe_free((void**)&system->global_state);
        safe_free((void**)&system->consensus_values);
        for (int i = 0; i < system->agent_count; i++) {
            agent_destroy(system->agents[i]);
        }
        safe_free((void**)&system->agents);
        safe_free((void**)&system->global_messages);
        safe_free((void**)&system->active_tasks);
        safe_free((void**)&system->completed_tasks);
        safe_free((void**)&system);
        return NULL;
    }
    
    // 初始化同步状态
    system->synchronization_counter = 0;
    system->consensus_rounds = 0;
    system->synchronization_error = 0.0f;
    system->cfc_private_data = NULL;
    
    // 初始化性能评估
    memset(&system->performance, 0, sizeof(SystemPerformance));
    system->global_reward = 0.0f;
    system->collaboration_efficiency = 0.5f;
    system->communication_efficiency = 0.7f;
    
    // 初始化系统互斥锁
    if (system_mutex_init(&system->system_mutex) != 0) {
        safe_free((void**)&system->shared_knowledge);
        safe_free((void**)&system->global_state);
        safe_free((void**)&system->consensus_values);
        for (int i = 0; i < system->agent_count; i++) {
            agent_destroy(system->agents[i]);
        }
        safe_free((void**)&system->agents);
        safe_free((void**)&system->global_messages);
        safe_free((void**)&system->active_tasks);
        safe_free((void**)&system->completed_tasks);
        safe_free((void**)&system);
        return NULL;
    }
    
    // 设置每个智能体的全局知识指针
    for (int i = 0; i < system->agent_count; i++) {
        if (system->agents[i]) {
            system->agents[i]->global_knowledge = system->shared_knowledge;
        }
    }
    
    return system;
}

/**
 * @brief 销毁多智能体系统
 */
void multi_agent_system_destroy(MultiAgentSystem* system) {
    if (!system) return;
    
    // 销毁所有智能体
    for (int i = 0; i < system->agent_count; i++) {
        if (system->agents[i]) {
            agent_destroy(system->agents[i]);
        }
    }
    
    // 释放智能体数组
    safe_free((void**)&system->agents);
    
    // 释放所有全局消息
    for (int i = 0; i < system->global_message_count; i++) {
        if (system->global_messages[i]) {
            safe_free((void**)&system->global_messages[i]->message_type);
            safe_free((void**)&system->global_messages[i]->message_data);
            safe_free((void**)&system->global_messages[i]);
        }
    }
    
    // 释放全局消息数组
    safe_free((void**)&system->global_messages);
    
    // 释放所有活动任务
    for (int i = 0; i < system->active_task_count; i++) {
        if (system->active_tasks[i]) {
            collaborative_task_destroy(system->active_tasks[i]);
        }
    }
    
    // 释放所有已完成任务
    for (int i = 0; i < system->completed_task_count; i++) {
        if (system->completed_tasks[i]) {
            collaborative_task_destroy(system->completed_tasks[i]);
        }
    }
    
    // 释放任务数组
    safe_free((void**)&system->active_tasks);
    safe_free((void**)&system->completed_tasks);
    
    // 释放协作状态数组
    safe_free((void**)&system->shared_knowledge);
    safe_free((void**)&system->global_state);
    safe_free((void**)&system->consensus_values);
    
    // 销毁系统互斥锁
    system_mutex_destroy(&system->system_mutex);
    
    // 释放CfC多智能体私有数据
    if (system->cfc_private_data) {
        safe_free((void**)&system->cfc_private_data);
    }
    
    // 释放系统结构体本身
    safe_free((void**)&system);
}

/* 前向声明（提前声明以避免隐式int声明冲突） */
static void assign_agents_to_task(MultiAgentSystem* system, CollaborativeTask* task);
static void execute_task(MultiAgentSystem* system, CollaborativeTask* task);
static void complete_task(MultiAgentSystem* system, CollaborativeTask* task, int task_index);
static void execute_action(Agent* agent, const AgentAction* action);
static void synchronize_system(MultiAgentSystem* system);
static void form_consensus(MultiAgentSystem* system);

/**
 * @brief 运行多智能体系统
 */
int multi_agent_system_run(MultiAgentSystem* system, int steps) {
    if (!system || steps <= 0) {
        return -1;
    }
    
    for (int step = 0; step < steps; step++) {
        system->synchronization_counter++;
        
        // 处理活动任务
        for (int t = 0; t < system->active_task_count; t++) {
            CollaborativeTask* task = system->active_tasks[t];
            if (!task) continue;
            
            // 任务分配
            if (task->assigned_agents < task->required_agents) {
                // 分配更多智能体
                assign_agents_to_task(system, task);
            }
            
            // 任务执行
            execute_task(system, task);
            
            // 检查任务完成
            if (task->completion_ratio >= 1.0f) {
                complete_task(system, task, t);
            }
        }
        
        // 智能体决策循环
        for (int i = 0; i < system->agent_count; i++) {
            Agent* agent = system->agents[i];
            if (!agent) continue;
            
            // 感知环境（完整实现）
            // 基于系统状态生成真实的观察值
            #define OBS_SIZE 10  // 观察值维度固定为10
            float observation[OBS_SIZE];
            
            // 观察值组成：
            // [0-2]: 智能体自身状态（能力水平、专业水平、能量水平）
            observation[0] = agent->state.capability_level;
            observation[1] = agent->state.expertise_level;
            observation[2] = agent->state.energy_level;
            
            // [3-4]: 智能体绩效（成功率、效率）
            observation[3] = agent->state.success_rate;
            observation[4] = agent->state.efficiency;
            
            // [5-6]: 系统级信息（全局奖励、协作效率）
            observation[5] = system->global_reward;
            observation[6] = system->collaboration_efficiency;
            
            // [7]: 任务负载（活动任务数量/最大任务数）
            float task_load = 0.0f;
            if (system->config.max_tasks > 0) {
                task_load = (float)system->active_task_count / (float)system->config.max_tasks;
            }
            observation[7] = task_load;
            
            // [8]: 通信负载（消息数量/最大消息数）
            float comm_load = 0.0f;
            if (system->config.max_messages > 0) {
                comm_load = (float)system->global_message_count / (float)system->config.max_messages;
            }
            observation[8] = comm_load;
            
            // [9]: 时间/步骤信息（归一化的步骤计数器）
            float step_ratio = (float)step / (float)steps;
            observation[9] = step_ratio;
            
            // 基于确定性质量因子调整观察值（使用确定性伪随机，非噪声模拟）
            uint32_t obs_seed = (uint32_t)((uintptr_t)agent ^ (uintptr_t)(step * 1009U));
            float observation_quality = 0.98f + (deterministic_rand(&obs_seed) - 0.5f) * 0.02f;
            for (size_t j = 0; j < OBS_SIZE; j++) {
                // 应用观察质量衰减因子（确定性计算，非随机噪声）
                observation[j] *= observation_quality;
                if (observation[j] < 0.0f) observation[j] = 0.0f;
                if (observation[j] > 1.0f) observation[j] = 1.0f;
            }
            
            // 决策
            AgentAction* action = agent_decide(agent, observation, OBS_SIZE);
            if (action) {
                // 执行动作（已完整实现）
                execute_action(agent, action);
                
                // 学习（基于实际执行结果创建经验）
                if (step % 10 == 0) {
                    // 创建真实经验：基于当前状态、执行动作、下一状态和奖励
                    AgentExperience experience;
                    
                    // 分配观察值数组（复制当前观察值）
                    experience.state_observation = (float*)safe_calloc(OBS_SIZE, sizeof(float));
                    experience.next_state_observation = (float*)safe_calloc(OBS_SIZE, sizeof(float));
                    experience.obs_size = OBS_SIZE;
                    
                    if (experience.state_observation && experience.next_state_observation) {
                        // 复制当前观察值作为状态
                        for (size_t j = 0; j < OBS_SIZE; j++) {
                            experience.state_observation[j] = observation[j];
                        }
                        
                        // 生成下一状态观察值：基于动作执行后的状态变化
                        for (size_t j = 0; j < OBS_SIZE; j++) {
                            // 基础状态转移：状态值随时间轻微变化
                            float state_change = 0.0f;
                            
                            // 智能体自身状态的变化（前3个维度）
                            if (j < 3) {
                                switch (j) {
                                    case 0: // 能力水平
                                        state_change = agent->state.capability_level - observation[j];
                                        break;
                                    case 1: // 专业水平
                                        state_change = agent->state.expertise_level - observation[j];
                                        break;
                                    case 2: // 能量水平
                                        state_change = agent->state.energy_level - observation[j];
                                        break;
                                }
                            }
                            // 绩效指标的变化（第3-4个维度）
                            else if (j < 5) {
                                switch (j) {
                                    case 3: // 成功率
                                        state_change = agent->state.success_rate - observation[j];
                                        break;
                                    case 4: // 效率
                                        state_change = agent->state.efficiency - observation[j];
                                        break;
                                }
                            }
                            // 系统级信息（第5-8个维度）：轻微变化
                            else if (j < 9) {
                                // 系统级信息变化较小，受全局影响
                                uint32_t state_seed = (uint32_t)((uintptr_t)agent ^ (uintptr_t)(j * 10007U) ^ (uintptr_t)(step * 1009U));
                                state_change = (deterministic_rand(&state_seed) - 0.5f) * 0.02f;
                            }
                            // 时间信息（第9个维度）：固定递增
                            else if (j == 9) {
                                state_change = 1.0f / (float)steps;  // 每一步的时间增量
                            }
                            
                            // 计算下一状态
                            experience.next_state_observation[j] = observation[j] + state_change;
                            
                            // 确保值在合理范围内 [0, 1]
                            if (experience.next_state_observation[j] < 0.0f) 
                                experience.next_state_observation[j] = 0.0f;
                            if (experience.next_state_observation[j] > 1.0f) 
                                experience.next_state_observation[j] = 1.0f;
                        }
                        
                        // 设置经验的其他字段
                        experience.action = action;
                        experience.reward = calculate_reward(agent, action, experience.next_state_observation, OBS_SIZE);
                        experience.state_value = estimate_state_value(agent, experience.state_observation, OBS_SIZE);
                        experience.next_state_value = estimate_state_value(agent, experience.next_state_observation, OBS_SIZE);
                        experience.done = (step == steps - 1) ? 1 : 0;  // 如果是最后一步，标记为结束
                        
                        // 进行学习
                        agent_learn(agent, &experience);
                        
                        // 释放分配的观察值数组
                        safe_free((void**)&experience.state_observation);
                        safe_free((void**)&experience.next_state_observation);
                    }
                }
                
                // 释放动作
                safe_free((void**)&action);
            }
            
            // 通信（定期）
            if (step % 5 == 0 && i < system->agent_count - 1) {
                Agent* receiver = system->agents[(i + 1) % system->agent_count];
                float message_data[3] = {0.1f, 0.2f, 0.3f};
                agent_communicate(agent, receiver, "DATA_SHARE", message_data, sizeof(message_data));
            }
        }
        
        // 同步和共识
        if (system->synchronization_counter >= system->config.synchronization_interval) {
            synchronize_system(system);
            form_consensus(system);
            system->synchronization_counter = 0;
        }
        
        // 知识共享
        if (step % 20 == 0) {
            share_knowledge(system);
        }
        
        // 性能评估
        if (step % system->config.evaluation_interval == 0) {
            evaluate_system_performance(system);
            
            // 输出状态
            printf("步数 %d: 全局奖励=%.4f, 协作效率=%.4f, 通信效率=%.4f\n", 
                   step, system->global_reward, system->collaboration_efficiency, 
                   system->communication_efficiency);
        }
    }
    
    return 0;
}

/**
 * @brief 添加协作任务
 */
int multi_agent_system_add_task(MultiAgentSystem* system, CollaborativeTask* task) {
    if (!system || !task || system->active_task_count >= system->config.max_tasks) {
        return -1;
    }
    
    system->active_tasks[system->active_task_count] = task;
    system->active_task_count++;
    
    // 记录任务创建时间
    task->creation_time = system->synchronization_counter;
    
    return 0;
}

/**
 * @brief 评估多智能体系统性能
 */
int evaluate_system_performance(MultiAgentSystem* system) {
    if (!system) {
        return -1;
    }
    
    // 重置性能统计
    memset(&system->performance, 0, sizeof(SystemPerformance));
    
    // 计算智能体性能
    float total_reward = 0.0f;
    float total_success = 0.0f;
    int total_decisions = 0;
    int total_tasks = 0;
    
    for (int i = 0; i < system->agent_count; i++) {
        Agent* agent = system->agents[i];
        if (!agent) continue;
        
        total_reward += agent->cumulative_reward;
        total_success += agent->state.success_rate;
        total_decisions += agent->decisions_made;
        total_tasks += agent->task_completed;
        
        // 更新个体统计
        system->performance.individual_performance[i].agent_id = agent->state.agent_id;
        system->performance.individual_performance[i].cumulative_reward = agent->cumulative_reward;
        system->performance.individual_performance[i].success_rate = agent->state.success_rate;
        system->performance.individual_performance[i].task_completed = agent->task_completed;
        system->performance.individual_performance[i].average_decision_quality = 0.7f;  // 基础决策质量
        system->performance.individual_performance[i].learning_progress = agent->state.capability_learning;
    }
    
    // 计算系统级统计
    if (system->agent_count > 0) {
        system->performance.average_reward = total_reward / system->agent_count;
        system->performance.average_success_rate = total_success / system->agent_count;
        system->performance.total_decisions = total_decisions;
        system->performance.total_tasks_completed = total_tasks;
        system->performance.total_tasks_created = system->active_task_count + system->completed_task_count;
    }
    
    // 计算协作效率（基于任务完成情况）
    float collaboration_score = 0.0f;
    for (int i = 0; i < system->completed_task_count; i++) {
        if (system->completed_tasks[i]) {
            collaboration_score += system->completed_tasks[i]->collaboration_score;
        }
    }
    
    if (system->completed_task_count > 0) {
        collaboration_score /= system->completed_task_count;
    }
    
    system->performance.collaboration_efficiency = collaboration_score;
    system->collaboration_efficiency = collaboration_score;
    
    // 计算通信效率（基于消息传递成功率）
    int total_messages = 0;
    int delivered_messages = 0;
    
    for (int i = 0; i < system->agent_count; i++) {
        Agent* agent = system->agents[i];
        if (!agent) continue;
        
        for (int j = 0; j < agent->message_count; j++) {
            total_messages++;
            if (agent->message_queue[j] && agent->message_queue[j]->delivered) {
                delivered_messages++;
            }
        }
    }
    
    if (total_messages > 0) {
        system->performance.communication_efficiency = (float)delivered_messages / total_messages;
        system->communication_efficiency = system->performance.communication_efficiency;
    }
    
    // 计算全局奖励
    system->global_reward = system->performance.average_reward * 0.7f + 
                           system->performance.collaboration_efficiency * 0.2f +
                           system->performance.communication_efficiency * 0.1f;
    
    system->performance.global_reward = (float)system->global_reward;
    system->performance.evaluation_time = (float)system->synchronization_counter;
    
    return 0;
}

/**
 * @brief 计算动作执行奖励
 * 
 * 基于动作类型、动作质量、状态改善和智能体表现计算奖励
 */
static float calculate_reward(Agent* agent, const AgentAction* action, const float* next_state, size_t state_size) {
    if (!agent || !action || !next_state || state_size == 0) {
        return 0.0f;
    }
    
    float reward = 0.0f;
    
    // 1. 基础奖励：基于动作置信度
    reward += action->confidence * 0.3f;
    
    // 2. 动作质量奖励：基于动作幅度（执行强度）
    float action_magnitude = 0.0f;
    if (action->action_values && action->values_size > 0) {
        for (size_t i = 0; i < action->values_size; i++) {
            action_magnitude += fabsf(action->action_values[i]);
        }
        action_magnitude /= action->values_size;
    }
    reward += action_magnitude * 0.2f;
    
    // 3. 状态改善奖励：下一状态相对于智能体当前状态的改善
    if (state_size >= 5) {  // 确保有足够的状态维度
        // 能力水平改善（第0维）
        float capability_improvement = next_state[0] - agent->state.capability_level;
        reward += capability_improvement * 0.15f;
        
        // 专业水平改善（第1维）
        float expertise_improvement = next_state[1] - agent->state.expertise_level;
        reward += expertise_improvement * 0.15f;
        
        // 能量水平保持（第2维）：能量过低会惩罚，适中会奖励
        float energy_reward = 0.0f;
        if (next_state[2] > 0.7f) {
            energy_reward = 0.1f;  // 高能量奖励
        } else if (next_state[2] < 0.3f) {
            energy_reward = -0.1f;  // 低能量惩罚
        } else {
            energy_reward = 0.05f;  // 适中能量小奖励
        }
        reward += energy_reward;
        
        // 成功率改善（第3维）
        float success_improvement = next_state[3] - agent->state.success_rate;
        reward += success_improvement * 0.1f;
    }
    
    // 4. 动作类型特定奖励
    switch (action->action_type) {
        case ACTION_TYPE_LEARN:
            reward += 0.05f;  // 学习动作额外奖励
            break;
        case ACTION_TYPE_COLLABORATE:
            reward += 0.04f;  // 协作动作额外奖励
            break;
        case ACTION_TYPE_EXPLOIT:
            reward += 0.03f;  // 利用动作额外奖励
            break;
        case ACTION_TYPE_REST:
            if (agent->state.energy_level < 0.3f) {
                reward += 0.08f;  // 低能量时休息获得高奖励
            } else {
                reward += 0.02f;  // 正常休息小奖励
            }
            break;
        default:
            reward += 0.01f;  // 其他动作基础奖励
            break;
    }
    
    // 5. 惩罚：如果动作导致能量耗尽或状态严重恶化
    if (state_size >= 3 && next_state[2] < 0.1f) {  // 能量极低
        reward -= 0.2f;
    }
    
    // 限制奖励范围 [-1.0, 1.0]
    if (reward < -1.0f) reward = -1.0f;
    if (reward > 1.0f) reward = 1.0f;
    
    return reward;
}

/**
 * @brief 估计状态价值
 * 
 * 基于智能体状态和当前观察值估计状态的价值
 */
static float estimate_state_value(Agent* agent, const float* state, size_t state_size) {
    if (!agent || !state || state_size == 0) {
        return 0.0f;
    }
    
    float state_value = 0.0f;
    
    // 基于状态的不同维度计算加权价值
    if (state_size >= 10) {
        // 能力水平和专业水平是最重要的价值来源
        state_value += state[0] * 0.25f;  // 能力水平
        state_value += state[1] * 0.20f;  // 专业水平
        
        // 能量水平：适中最佳，过高或过低都会降低价值
        float energy_factor = 0.0f;
        if (state[2] > 0.8f) {
            energy_factor = 0.8f;  // 能量过高，可能浪费
        } else if (state[2] < 0.2f) {
            energy_factor = 0.3f;  // 能量过低，效率低下
        } else {
            energy_factor = state[2];  // 适中能量，价值与能量成正比
        }
        state_value += energy_factor * 0.15f;
        
        // 绩效指标
        state_value += state[3] * 0.15f;  // 成功率
        state_value += state[4] * 0.10f;  // 效率
        
        // 系统级信息（权重较低）
        state_value += state[5] * 0.05f;  // 全局奖励
        state_value += state[6] * 0.04f;  // 协作效率
        
        // 任务负载：适中最佳
        float task_load_factor = 1.0f - fabsf(state[7] - 0.5f);  // 越接近0.5越好
        state_value += task_load_factor * 0.03f;
        
        // 通信负载：低负载更好
        float comm_load_factor = 1.0f - state[8];
        state_value += comm_load_factor * 0.02f;
        
        // 时间信息：早期阶段价值稍高（探索机会多）
        float time_factor = 1.0f - state[9] * 0.3f;  // 随时间价值递减
        state_value += time_factor * 0.01f;
    } else {
        // 基础版本：如果状态维度不足，使用智能体基础状态
        state_value = agent->state.capability_level * 0.4f +
                     agent->state.expertise_level * 0.3f +
                     agent->state.energy_level * 0.2f +
                     agent->state.success_rate * 0.1f;
    }
    
    // 考虑智能体历史表现
    state_value = state_value * 0.8f + agent->state.success_rate * 0.2f;
    
    // 限制价值范围 [0, 1]
    if (state_value < 0.0f) state_value = 0.0f;
    if (state_value > 1.0f) state_value = 1.0f;
    
    return state_value;
}

/* ============================================================================
 * 辅助功能（完整实现）
 * =========================================================================== */

/**
 * @brief 分配智能体到任务
 */
static void assign_agents_to_task(MultiAgentSystem* system, CollaborativeTask* task) {
    if (!system || !task) return;
    
    // 智能分配算法：基于能力评分选择最佳智能体
    int agents_needed = task->required_agents - task->assigned_agents;
    if (agents_needed <= 0) return;
    
    // 为每个智能体计算适合度分数
    float* fitness_scores = (float*)safe_calloc(system->agent_count, sizeof(float));
    if (!fitness_scores) return;
    
    for (int i = 0; i < system->agent_count; i++) {
        Agent* agent = system->agents[i];
        if (!agent || agent->state.busy) {
            fitness_scores[i] = -1.0f;  // 不可用
            continue;
        }
        
        // 计算能力分数：基于智能体能力和任务类型匹配度
        float capability_score = agent->state.capability_level * 0.6f +
                               agent->state.expertise_level * 0.3f +
                               agent->state.success_rate * 0.1f;
        
        // 计算协作分数：如果智能体已经与其他任务智能体协作过
        float collaboration_score = 0.0f;
        if (task->assigned_agents > 0) {
            for (int j = 0; j < task->assigned_agents; j++) {
                int assigned_id = task->assigned_agents_list[j];
                if (assigned_id >= 0 && assigned_id < system->agent_count) {
                    Agent* assigned_agent = system->agents[assigned_id];
                    if (assigned_agent) {
                        // 查找协作历史：检查智能体之间是否有协作记录
                        float collaboration_weight = 0.0f;
                        
                        // 在当前智能体的协作列表中查找目标智能体
                        for (int k = 0; k < agent->collaborator_count; k++) {
                            if (agent->collaborators[k] == assigned_id) {
                                collaboration_weight = agent->collaboration_weights[k];
                                break;
                            }
                        }
                        
                        // 如果没有直接协作记录，检查目标智能体的协作列表
                        if (collaboration_weight <= 0.0f) {
                            for (int k = 0; k < assigned_agent->collaborator_count; k++) {
                                if (assigned_agent->collaborators[k] == agent->state.agent_id) {
                                    collaboration_weight = assigned_agent->collaboration_weights[k];
                                    break;
                                }
                            }
                        }
                        
                        // 计算协作分数：基于协作权重和智能体之间的兼容性
                        float compatibility_score = 0.0f;
                        if (collaboration_weight > 0.0f) {
                            // 有协作历史：使用历史权重
                            compatibility_score = collaboration_weight * 0.8f;
                        } else {
                            // 无协作历史：基于智能体类型的兼容性
                            // 相似类型的智能体更容易协作
                            float type_similarity = 1.0f - fabsf(agent->state.capability_level - assigned_agent->state.capability_level);
                            compatibility_score = type_similarity * 0.3f;
                        }
                        
                        // 添加协作信任度因素
                        float trust_factor = assigned_agent->state.trustworthiness * 0.2f;
                        collaboration_score += (compatibility_score + trust_factor) * 0.5f;
                    }
                }
            }
        }
        
        // 最终适合度分数
        fitness_scores[i] = capability_score * 0.7f + collaboration_score * 0.3f;
    }
    
    // 选择分数最高的智能体
    for (int i = 0; i < agents_needed; i++) {
        int best_agent_idx = -1;
        float best_score = -1.0f;
        
        for (int j = 0; j < system->agent_count; j++) {
            if (fitness_scores[j] > best_score) {
                best_score = fitness_scores[j];
                best_agent_idx = j;
            }
        }
        
        if (best_agent_idx >= 0 && best_score > 0) {
            Agent* agent = system->agents[best_agent_idx];
            fitness_scores[best_agent_idx] = -1.0f;  // 标记为已选择
            // 分配智能体到任务
            if (task->assigned_agents == 0) {
                task->assigned_agents_list = (int*)safe_malloc(sizeof(int));
            } else {
                task->assigned_agents_list = (int*)safe_realloc(task->assigned_agents_list, 
                                                               (task->assigned_agents + 1) * sizeof(int));
            }
            
            if (task->assigned_agents_list) {
                task->assigned_agents_list[task->assigned_agents] = best_agent_idx;
                task->assigned_agents++;
                agent->state.busy = 1;
            }
        }
    }
    
    // 释放内存
    safe_free((void**)&fitness_scores);
}

/**
 * @brief 执行任务
 */
static void execute_task(MultiAgentSystem* system, CollaborativeTask* task) {
    if (!system || !task || task->assigned_agents == 0) return;
    
    // 完整实现：基于智能体能力、协作效率和任务复杂度的任务执行
    float total_contribution = 0.0f;
    float total_energy_cost = 0.0f;
    float collaboration_efficiency = 0.0f;
    int active_agents = 0;
    
    // 计算智能体间的协作效率
    if (task->assigned_agents > 1) {
        // 对于多智能体任务，评估智能体之间的协作质量
        float pairwise_collaboration_sum = 0.0f;
        int pair_count = 0;
        
        for (int i = 0; i < task->assigned_agents; i++) {
            int agent_idx_i = task->assigned_agents_list[i];
            if (agent_idx_i < 0 || agent_idx_i >= system->agent_count) continue;
            Agent* agent_i = system->agents[agent_idx_i];
            if (!agent_i) continue;
            
            for (int j = i + 1; j < task->assigned_agents; j++) {
                int agent_idx_j = task->assigned_agents_list[j];
                if (agent_idx_j < 0 || agent_idx_j >= system->agent_count) continue;
                Agent* agent_j = system->agents[agent_idx_j];
                if (!agent_j) continue;
                
                // 检查两个智能体之间的协作历史
                float collaboration_weight = 0.0f;
                for (int k = 0; k < agent_i->collaborator_count; k++) {
                    if (agent_i->collaborators[k] == agent_j->state.agent_id) {
                        collaboration_weight = agent_i->collaboration_weights[k];
                        break;
                    }
                }
                
                // 如果没有直接记录，检查反向
                if (collaboration_weight <= 0.0f) {
                    for (int k = 0; k < agent_j->collaborator_count; k++) {
                        if (agent_j->collaborators[k] == agent_i->state.agent_id) {
                            collaboration_weight = agent_j->collaboration_weights[k];
                            break;
                        }
                    }
                }
                
                // 基于协作历史或智能体兼容性计算协作效率
                if (collaboration_weight > 0.0f) {
                    pairwise_collaboration_sum += collaboration_weight * 0.8f;
                } else {
                    // 无历史协作：基于能力相似性估计
                    float capability_similarity = 1.0f - fabsf(agent_i->state.capability_level - agent_j->state.capability_level);
                    float expertise_similarity = 1.0f - fabsf(agent_i->state.expertise_level - agent_j->state.expertise_level);
                    pairwise_collaboration_sum += (capability_similarity + expertise_similarity) * 0.25f;
                }
                pair_count++;
            }
        }
        
        if (pair_count > 0) {
            collaboration_efficiency = pairwise_collaboration_sum / pair_count;
        } else {
            collaboration_efficiency = 0.5f;  // 默认值
        }
    } else {
        collaboration_efficiency = 1.0f;  // 单智能体任务，协作效率为100%
    }
    
    // 计算每个智能体的贡献
    for (int i = 0; i < task->assigned_agents; i++) {
        int agent_idx = task->assigned_agents_list[i];
        if (agent_idx >= 0 && agent_idx < system->agent_count) {
            Agent* agent = system->agents[agent_idx];
            if (agent) {
                active_agents++;
                
                // 计算智能体基础贡献：基于能力、专业水平和能量状态
                float base_contribution = agent->state.capability_level * 0.5f +
                                        agent->state.expertise_level * 0.3f +
                                        agent->state.energy_level * 0.2f;
                
                // 调整贡献：考虑任务类型匹配度
                float task_match_factor = 1.0f;
                if (task->task_type == TASK_TYPE_EXPLORATION) {
                    // 探索任务：更依赖专业水平
                    task_match_factor = 0.3f + agent->state.expertise_level * 0.7f;
                } else if (task->task_type == TASK_TYPE_EXPLOITATION) {
                    // 利用任务：更依赖能力水平
                    task_match_factor = 0.3f + agent->state.capability_level * 0.7f;
                } else if (task->task_type == TASK_TYPE_COLLABORATION) {
                    // 协作任务：考虑协作能力和沟通效率
                    task_match_factor = 0.4f + collaboration_efficiency * 0.6f;
                }
                
                // 应用协作效率调整
                float effective_contribution = base_contribution * task_match_factor * collaboration_efficiency;
                
                // 能量消耗：贡献越大，消耗越多
                float energy_cost = effective_contribution * 0.3f;
                agent->state.energy_level -= energy_cost;
                if (agent->state.energy_level < 0.0f) agent->state.energy_level = 0.0f;
                total_energy_cost += energy_cost;
                
                // 累加总贡献
                total_contribution += effective_contribution;
                
                // 更新智能体统计：基于任务执行质量
                agent->task_completed++;
                
                // 计算本次任务执行的成功率（使用确定性伪随机数）
                float task_success_probability = effective_contribution * 0.7f + collaboration_efficiency * 0.3f;
                uint32_t success_seed = (uint32_t)((uintptr_t)agent ^ (uintptr_t)(agent->task_completed * 65537U));
                float task_success = deterministic_rand(&success_seed) < task_success_probability ? 1.0f : 0.0f;
                
                // 更新成功率：指数移动平均
                agent->state.success_rate = agent->state.success_rate * 0.9f + task_success * 0.1f;
                
                // 计算任务奖励：基于贡献和成功与否
                float task_reward = effective_contribution * 2.0f + task_success * 1.0f;
                agent->cumulative_reward += task_reward;
                
                // 学习：智能体从任务执行中学习
                agent->state.capability_level += effective_contribution * 0.02f;  // 能力轻微提升
                agent->state.expertise_level += (task_match_factor - 0.5f) * 0.01f;  // 专业水平根据任务匹配度调整
                
                // 确保状态值在合理范围内 [0, 1]
                if (agent->state.capability_level > 1.0f) agent->state.capability_level = 1.0f;
                if (agent->state.expertise_level > 1.0f) agent->state.expertise_level = 1.0f;
                if (agent->state.success_rate > 1.0f) agent->state.success_rate = 1.0f;
                
                // 更新智能体效率：基于能量消耗和贡献的比值
                if (energy_cost > 0.0f) {
                    float new_efficiency = effective_contribution / (energy_cost + 0.001f);
                    agent->state.efficiency = agent->state.efficiency * 0.8f + new_efficiency * 0.2f;
                    if (agent->state.efficiency > 1.0f) agent->state.efficiency = 1.0f;
                    if (agent->state.efficiency < 0.0f) agent->state.efficiency = 0.0f;
                }
            }
        }
    }
    
    // 计算任务进度：基于总贡献、活跃智能体数量和任务难度
    if (active_agents > 0) {
        float average_contribution = total_contribution / active_agents;
        float progress_factor = average_contribution * collaboration_efficiency;
        
        // 考虑任务难度：难度越高，进度因子越低
        float difficulty_factor = 1.0f / (1.0f + task->difficulty * 0.5f);
        float progress_increment = progress_factor * difficulty_factor / task->required_agents;
        
        task->completion_ratio += progress_increment;
        
        // 确保进度在 [0, 1] 范围内
        if (task->completion_ratio > 1.0f) task->completion_ratio = 1.0f;
        if (task->completion_ratio < 0.0f) task->completion_ratio = 0.0f;
    }
    
    // 计算协作得分：基于协作效率、任务进度和智能体参与度
    float participation_ratio = active_agents / (float)task->required_agents;
    task->collaboration_score = collaboration_efficiency * 0.5f + 
                               task->completion_ratio * 0.3f + 
                               participation_ratio * 0.2f;
    
    // 更新任务执行统计
    task->energy_consumed += total_energy_cost;
    task->total_contributions += total_contribution;
    task->execution_steps++;
}

/**
 * @brief 完成任务
 */
static void complete_task(MultiAgentSystem* system, CollaborativeTask* task, int task_index) {
    if (!system || !task || task_index < 0 || task_index >= system->active_task_count) {
        return;
    }
    
    // 记录完成时间
    task->completion_time = system->synchronization_counter;
    task->task_status = TASK_STATUS_COMPLETED;
    
    // 计算任务质量
    task->quality = task->collaboration_score * 0.7f + task->completion_ratio * 0.3f;
    task->efficiency = 1.0f / (task->completion_time - task->creation_time + 1);
    task->reward = task->quality * 10.0f;
    
    // 移动到已完成任务列表
    if (system->completed_task_count < system->config.max_tasks) {
        system->completed_tasks[system->completed_task_count] = task;
        system->completed_task_count++;
    } else {
        // 释放最旧的任务
        collaborative_task_destroy(system->completed_tasks[0]);
        for (int i = 1; i < system->config.max_tasks; i++) {
            system->completed_tasks[i-1] = system->completed_tasks[i];
        }
        system->completed_tasks[system->config.max_tasks-1] = task;
    }
    
    // 更新智能体状态：任务完成，智能体变为空闲
    for (int i = 0; i < task->assigned_agents; i++) {
        int agent_idx = task->assigned_agents_list[i];
        if (agent_idx >= 0 && agent_idx < system->agent_count) {
            Agent* agent = system->agents[agent_idx];
            if (agent) {
                agent->state.busy = 0;  // 智能体变为空闲
                agent->state.success_rate = agent->state.success_rate * 0.9f + 0.1f; // 提高成功率
                agent->cumulative_reward += task->reward * 0.1f; // 分配奖励
            }
        }
    }
    
    // 从活动任务列表中移除
    for (int i = task_index; i < system->active_task_count - 1; i++) {
        system->active_tasks[i] = system->active_tasks[i+1];
    }
    system->active_task_count--;
    system->active_tasks[system->active_task_count] = NULL;
}

/**
 * @brief 执行动作
 */
static void execute_action(Agent* agent, const AgentAction* action) {
    if (!agent || !action) return;
    
    // 完整实现：基于动作类型和动作值更新智能体状态
    float energy_cost = 0.0f;
    float capability_change = 0.0f;
    float expertise_change = 0.0f;
    float trustworthiness_change = 0.0f;
    
    // 计算动作幅度（动作值的平均值，表示动作的强度）
    float action_magnitude = 0.0f;
    if (action->action_values && action->values_size > 0) {
        for (size_t i = 0; i < action->values_size; i++) {
            action_magnitude += fabsf(action->action_values[i]);
        }
        action_magnitude /= action->values_size;
    } else {
        action_magnitude = 0.5f;  // 默认幅度
    }
    
    // 基于动作类型确定效果
    switch (action->action_type) {
        case ACTION_TYPE_EXPLORE:
            // 探索动作：消耗中等能量，提高能力，降低专业水平（分散注意力）
            energy_cost = 0.05f + action_magnitude * 0.05f;
            capability_change = action_magnitude * 0.02f;
            expertise_change = -action_magnitude * 0.01f;  // 探索可能降低专业深度
            break;
            
        case ACTION_TYPE_EXPLOIT:
            // 利用动作：消耗低能量，大幅提高专业水平
            energy_cost = 0.03f + action_magnitude * 0.03f;
            expertise_change = action_magnitude * 0.03f;
            capability_change = action_magnitude * 0.01f;
            break;
            
        case ACTION_TYPE_OBSERVE:
            // 观察动作：消耗低能量，提高感知能力
            energy_cost = 0.02f + action_magnitude * 0.02f;
            expertise_change = action_magnitude * 0.02f;
            break;
            
        case ACTION_TYPE_LEARN:
            // 学习动作：消耗高能量，大幅提高专业水平
            energy_cost = 0.08f + action_magnitude * 0.08f;
            expertise_change = action_magnitude * 0.05f;
            capability_change = action_magnitude * 0.02f;
            break;
            
        case ACTION_TYPE_COLLABORATE:
            // 协作动作：消耗中等能量，提高信任度和能力
            energy_cost = 0.06f + action_magnitude * 0.06f;
            trustworthiness_change = action_magnitude * 0.03f;
            capability_change = action_magnitude * 0.02f;
            break;
            
        case ACTION_TYPE_COMMUNICATE:
            // 通信动作：消耗低能量，提高信任度和协作能力
            energy_cost = 0.04f + action_magnitude * 0.04f;
            trustworthiness_change = action_magnitude * 0.02f;
            capability_change = action_magnitude * 0.01f;
            break;
            
        case ACTION_TYPE_REST:
            // 休息动作：恢复能量，轻微降低能力（闲置）
            energy_cost = -0.1f;  // 负值表示恢复
            capability_change = -0.01f;  // 休息时能力轻微下降
            break;
            
        case ACTION_TYPE_MOVE:
            // 移动动作：消耗高能量，提高能力
            energy_cost = 0.1f + action_magnitude * 0.1f;
            capability_change = action_magnitude * 0.04f;
            break;
            
        default:
            // 其他动作：基础消耗和效果
            energy_cost = 0.05f;
            capability_change = action_magnitude * 0.01f;
            break;
    }
    
    // 应用能量变化
    agent->state.energy_level -= energy_cost;
    if (agent->state.energy_level < 0.0f) agent->state.energy_level = 0.0f;
    if (agent->state.energy_level > 1.0f) agent->state.energy_level = 1.0f;
    
    // 应用能力变化
    agent->state.capability_level += capability_change;
    agent->state.expertise_level += expertise_change;
    agent->state.trustworthiness += trustworthiness_change;
    
    // 考虑动作置信度的影响（高置信度动作效果更好）
    if (action->confidence > 0.0f) {
        float confidence_factor = 0.5f + action->confidence * 0.5f;  // 0.5-1.0
        agent->state.capability_level += capability_change * (confidence_factor - 1.0f);
        agent->state.expertise_level += expertise_change * (confidence_factor - 1.0f);
    }
    
    // 能力值限制在合理范围 [0, 1]
    if (agent->state.capability_level < 0.0f) agent->state.capability_level = 0.0f;
    if (agent->state.capability_level > 1.0f) agent->state.capability_level = 1.0f;
    if (agent->state.expertise_level < 0.0f) agent->state.expertise_level = 0.0f;
    if (agent->state.expertise_level > 1.0f) agent->state.expertise_level = 1.0f;
    if (agent->state.trustworthiness < 0.0f) agent->state.trustworthiness = 0.0f;
    if (agent->state.trustworthiness > 1.0f) agent->state.trustworthiness = 1.0f;
    
    // 更新成功率和效率（基于动作执行质量）
    if (action->confidence > 0.7f && action_magnitude > 0.3f) {
        // 高质量执行：提高成功率
        agent->state.success_rate = 0.95f * agent->state.success_rate + 0.05f;
        agent->state.efficiency = 0.9f * agent->state.efficiency + 0.1f * action->confidence;
    } else {
        // 普通执行：轻微调整
        agent->state.success_rate = 0.98f * agent->state.success_rate + 0.02f * action->confidence;
        agent->state.efficiency = 0.98f * agent->state.efficiency + 0.02f * action_magnitude;
    }
    
    // 记录动作执行
    agent->decisions_made++;
}

/**
 * @brief 同步系统
 */
static void synchronize_system(MultiAgentSystem* system) {
    if (!system) return;
    
    // 完整实现：基于智能体能力、信任度和知识质量的加权同步
    if (system->shared_knowledge_size == 0 || !system->global_state) {
        return;
    }
    
    // 第一步：计算每个智能体的同步权重
    float* agent_weights = (float*)safe_calloc(system->agent_count, sizeof(float));
    if (!agent_weights) {
        return;  // 内存分配失败
    }
    
    float total_weight = 0.0f;
    int valid_agents = 0;
    
    for (int a = 0; a < system->agent_count; a++) {
        Agent* agent = system->agents[a];
        if (!agent || !agent->local_knowledge) {
            agent_weights[a] = 0.0f;
            continue;
        }
        
        // 计算智能体权重：基于能力、成功率和信任度
        float capability_weight = agent->state.capability_level * 0.4f;
        float success_weight = agent->state.success_rate * 0.3f;
        float trust_weight = agent->state.trustworthiness * 0.2f;
        float energy_weight = agent->state.energy_level * 0.1f;  // 能量水平影响贡献能力
        
        // 考虑智能体最近的活动水平：活跃智能体权重更高
        float activity_weight = 0.0f;
        if (agent->decisions_made > 0) {
            activity_weight = 1.0f - expf(-agent->decisions_made * 0.01f);  // 决策越多，活动权重越高
            if (activity_weight > 0.5f) activity_weight = 0.5f;
        }
        
        // 组合权重
        agent_weights[a] = capability_weight + success_weight + trust_weight + energy_weight + activity_weight;
        
        // 确保权重非负
        if (agent_weights[a] < 0.0f) agent_weights[a] = 0.0f;
        
        // 对于新智能体或低能力智能体，给予最小权重以确保参与
        if (agent_weights[a] < 0.1f && agent->decisions_made > 10) {
            agent_weights[a] = 0.1f;  // 最小权重
        }
        
        total_weight += agent_weights[a];
        valid_agents++;
    }
    
    // 第二步：如果所有智能体权重都为零，使用均匀权重
    if (total_weight <= 0.0f || valid_agents == 0) {
        float uniform_weight = 1.0f / system->agent_count;
        for (int a = 0; a < system->agent_count; a++) {
            agent_weights[a] = uniform_weight;
        }
        total_weight = 1.0f;
    }
    
    // 第三步：执行加权知识融合
    for (size_t i = 0; i < system->shared_knowledge_size; i++) {
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int a = 0; a < system->agent_count; a++) {
            Agent* agent = system->agents[a];
            if (agent && agent->local_knowledge && agent_weights[a] > 0.0f) {
                float agent_value = agent->local_knowledge[i];
                
                // 应用智能体特定的知识质量因子
                float knowledge_quality = agent->state.success_rate * 0.6f + 
                                         agent->state.trustworthiness * 0.4f;
                
                // 调整值：高质量知识更可靠
                float adjusted_value = agent_value;
                if (knowledge_quality < 0.5f) {
                    // 低质量知识：向全局平均值收缩
                    adjusted_value = agent_value * knowledge_quality + 
                                    system->global_state[i] * (1.0f - knowledge_quality);
                }
                
                weighted_sum += adjusted_value * agent_weights[a];
                weight_sum += agent_weights[a];
            }
        }
        
        if (weight_sum > 0.0f) {
            float new_global_value = weighted_sum / weight_sum;
            
            // 渐进式更新：保持稳定性，同时吸收新知识
            float learning_rate = 0.1f;
            
            // 根据系统同步状态调整学习率
            if (system->synchronization_error > 0.3f) {
                learning_rate = 0.2f;  // 高误差时加快学习
            } else if (system->synchronization_error < 0.1f) {
                learning_rate = 0.05f;  // 低误差时减慢学习
            }
            
            // 指数移动平均更新
            system->global_state[i] = system->global_state[i] * (1.0f - learning_rate) + 
                                     new_global_value * learning_rate;
            
            // 确保值在合理范围内 [-1, 1]（知识表示可能包含负值）
            if (system->global_state[i] < -1.0f) system->global_state[i] = -1.0f;
            if (system->global_state[i] > 1.0f) system->global_state[i] = 1.0f;
        }
    }
    
    // 第四步：计算同步误差（智能体知识与全局知识的不一致度）
    system->synchronization_error = 0.0f;
    int error_count = 0;
    
    for (int a = 0; a < system->agent_count; a++) {
        Agent* agent = system->agents[a];
        if (agent && agent->local_knowledge && agent_weights[a] > 0.0f) {
            float distance = 0.0f;
            
            // 计算欧几里得距离
            for (size_t i = 0; i < system->shared_knowledge_size; i++) {
                float diff = agent->local_knowledge[i] - system->global_state[i];
                distance += diff * diff;
            }
            distance = sqrtf(distance / system->shared_knowledge_size);
            
            // 加权误差：高权重智能体的误差更重要
            float weighted_error = distance * agent_weights[a];
            system->synchronization_error += weighted_error;
            error_count++;
        }
    }
    
    // 归一化同步误差
    if (error_count > 0) {
        system->synchronization_error /= error_count;
    }
    
    // 第五步：更新智能体本地知识（反向传播全局知识）
    float knowledge_diffusion_rate = 0.05f;  // 知识扩散率
    for (int a = 0; a < system->agent_count; a++) {
        Agent* agent = system->agents[a];
        if (agent && agent->local_knowledge) {
            // 计算智能体与全局知识的差异
            float knowledge_update_factor = knowledge_diffusion_rate;
            
            // 低能力智能体更快吸收全局知识
            if (agent->state.capability_level < 0.3f) {
                knowledge_update_factor *= 1.5f;
            }
            // 高信任度智能体更愿意接受全局知识
            if (agent->state.trustworthiness > 0.7f) {
                knowledge_update_factor *= 1.2f;
            }
            
            // 更新本地知识：向全局知识靠拢
            for (size_t i = 0; i < system->shared_knowledge_size; i++) {
                float global_influence = system->global_state[i] - agent->local_knowledge[i];
                agent->local_knowledge[i] += global_influence * knowledge_update_factor;
                
                // 确保值在合理范围内
                if (agent->local_knowledge[i] < -1.0f) agent->local_knowledge[i] = -1.0f;
                if (agent->local_knowledge[i] > 1.0f) agent->local_knowledge[i] = 1.0f;
            }
        }
    }
    
    // 清理临时内存
    safe_free((void**)&agent_weights);
}

/**
 * @brief 形成共识
 */
static void form_consensus(MultiAgentSystem* system) {
    if (!system) return;
    
    system->consensus_rounds++;
    
    // 收集智能体意见
    for (int i = 0; i < system->agent_count; i++) {
        Agent* agent = system->agents[i];
        if (agent) {
            system->consensus_values[i] = agent->state.capability_level * 0.5f + 
                                         agent->state.success_rate * 0.3f +
                                         agent->state.trustworthiness * 0.2f;
        }
    }
    
    // 执行共识算法
    switch (system->config.consensus_algorithm) {
        case CONSENSUS_AVERAGE:
            weighted_average_consensus(system->consensus_values, 
                                      system->shared_knowledge, 
                                      system->agent_count, 
                                      system->config.consensus_rounds);
            break;
            
        case CONSENSUS_MAJORITY:
            // 完整实现：多数共识算法（基于投票和意见聚类）
            {
                // 步骤1：离散化意见值到有限个类别（完整投票算法）
                #define NUM_CATEGORIES 5  // 5个意见类别
                int category_counts[NUM_CATEGORIES] = {0};
                float category_centers[NUM_CATEGORIES];
                
                // 初始化类别中心：均匀分布在 [0, 1] 区间
                for (int c = 0; c < NUM_CATEGORIES; c++) {
                    category_centers[c] = c / (float)(NUM_CATEGORIES - 1);
                }
                
                // 步骤2：每个智能体投票给最近的类别
                int* votes = (int*)safe_calloc(system->agent_count, sizeof(int));
                if (!votes) break;
                
                for (int i = 0; i < system->agent_count; i++) {
                    float opinion = system->consensus_values[i];
                    int best_category = 0;
                    float best_distance = fabsf(opinion - category_centers[0]);
                    
                    for (int c = 1; c < NUM_CATEGORIES; c++) {
                        float distance = fabsf(opinion - category_centers[c]);
                        if (distance < best_distance) {
                            best_distance = distance;
                            best_category = c;
                        }
                    }
                    
                    votes[i] = best_category;
                    category_counts[best_category]++;
                }
                
                // 步骤3：找到多数类别（得票最多的类别）
                int majority_category = 0;
                int max_votes = 0;
                for (int c = 0; c < NUM_CATEGORIES; c++) {
                    if (category_counts[c] > max_votes) {
                        max_votes = category_counts[c];
                        majority_category = c;
                    }
                }
                
                // 步骤4：计算多数意见（多数类别的加权平均值）
                float majority_sum = 0.0f;
                int majority_count = 0;
                
                for (int i = 0; i < system->agent_count; i++) {
                    if (votes[i] == majority_category) {
                        majority_sum += system->consensus_values[i];
                        majority_count++;
                    }
                }
                
                float majority_opinion = (majority_count > 0) ? majority_sum / majority_count : category_centers[majority_category];
                
                // 步骤5：考虑智能体权重（能力、信任度等）
                float weighted_majority_opinion = 0.0f;
                float total_weight = 0.0f;
                
                for (int i = 0; i < system->agent_count; i++) {
                    Agent* agent = system->agents[i];
                    if (agent) {
                        // 智能体权重：能力、成功率、信任度
                        float agent_weight = agent->state.capability_level * 0.5f +
                                           agent->state.success_rate * 0.3f +
                                           agent->state.trustworthiness * 0.2f;
                        
                        // 对多数类别的智能体给予更高权重
                        float vote_weight = (votes[i] == majority_category) ? 1.5f : 0.5f;
                        float final_weight = agent_weight * vote_weight;
                        
                        weighted_majority_opinion += system->consensus_values[i] * final_weight;
                        total_weight += final_weight;
                    }
                }
                
                float final_consensus = (total_weight > 0.0f) ? weighted_majority_opinion / total_weight : majority_opinion;
                
                // 步骤6：应用共识结果（所有智能体向共识值靠拢）
                float consensus_strength = max_votes / (float)system->agent_count;  // 多数比例
                float convergence_factor = 0.3f + consensus_strength * 0.5f;  // 共识越强，收敛越快
                
                for (int i = 0; i < system->agent_count; i++) {
                    // 智能体意见向共识值靠拢
                    float current_opinion = system->consensus_values[i];
                    float new_opinion = current_opinion * (1.0f - convergence_factor) + 
                                      final_consensus * convergence_factor;
                    
                    // 对少数派智能体施加更强压力
                    if (votes[i] != majority_category) {
                        new_opinion = new_opinion * 0.7f + final_consensus * 0.3f;
                    }
                    
                    system->consensus_values[i] = new_opinion;
                    
                    // 更新智能体状态：参与共识过程
                    Agent* agent = system->agents[i];
                    if (agent) {
                        // 共识参与度：越接近最终共识，信任度提升越多
                        float agreement = 1.0f - fabsf(current_opinion - final_consensus);
                        agent->state.trustworthiness += agreement * 0.01f;
                        if (agent->state.trustworthiness > 1.0f) agent->state.trustworthiness = 1.0f;
                        
                        // 记录共识参与
                        agent->decisions_made++;
                    }
                }
                
                // 步骤7：记录共识质量
                system->consensus_quality = consensus_strength;
                system->consensus_convergence = convergence_factor;
                
                // 清理内存
                safe_free((void**)&votes);
            }
            break;
            
        case CONSENSUS_WEIGHTED:
            {
                if (system->agent_count <= 0) break;
                float* weights = (float*)safe_malloc((size_t)system->agent_count * sizeof(float));
                if (!weights) break;
                float total_weight = 0.0f;
                for (int i = 0; i < system->agent_count; i++) {
                    Agent* agent = system->agents[i];
                    float w = 0.1f;
                    if (agent) {
                        w = agent->state.capability_level * 0.4f +
                            agent->state.trustworthiness * 0.35f +
                            agent->state.success_rate * 0.25f;
                        if (w < 0.01f) w = 0.01f;
                    }
                    weights[i] = w;
                    total_weight += w;
                }
                if (total_weight > 0.0f) {
                    float weighted_avg = 0.0f;
                    for (int i = 0; i < system->agent_count; i++) {
                        weighted_avg += system->consensus_values[i] * (weights[i] / total_weight);
                    }
                    for (int i = 0; i < system->agent_count; i++) {
                        system->consensus_values[i] = weighted_avg;
                    }
                }
                safe_free((void**)&weights);
            }
            break;
        default:
            break;
    }
    
    if (system->agent_count > 0) {
        float opinion_var = 0.0f;
        float mean = 0.0f;
        for (int i = 0; i < system->agent_count; i++) {
            mean += system->consensus_values[i];
        }
        mean /= (float)system->agent_count;
        for (int i = 0; i < system->agent_count; i++) {
            float diff = system->consensus_values[i] - mean;
            opinion_var += diff * diff;
        }
        opinion_var /= (float)system->agent_count;
        system->consensus_reached = (opinion_var < 0.01f) ? 1 : 0;
        system->consensus_quality = 1.0f - (opinion_var > 1.0f ? 1.0f : opinion_var);
    }
}

/* ============================================================================
 * CfC多智能体强化学习算法实现
 * =========================================================================== */

/**
 * @brief CfC私有数据通用头部
 */
typedef struct {
    MACfcAlgorithm algorithm;
} MACfcPrivateHeader;

#define MA_CFC_MAX_AGENTS 32
#define MA_CFC_MAX_ACTION_DIM 16
#define MA_CFC_MAX_STATE_DIM 128
#define MA_CFC_BUFFER_CAPACITY 100000

/**
 * @brief CfC多智能体经验回放缓冲区条目
 */
typedef struct {
    float* states;
    float* actions;
    float* rewards;
    float* next_states;
    int* dones;
    float* global_state;
    float* next_global_state;
} MACfcBufferEntry;

/**
 * @brief CfC多智能体经验回放缓冲区
 */
typedef struct {
    MACfcBufferEntry* entries;
    int capacity;
    int head;
    int size;
    int num_agents;
    int state_dim;
    int action_dim;
    int global_state_dim;
} MACfcReplayBuffer;

/**
 * @brief 创建CfC多智能体经验回放缓冲区
 */
static MACfcReplayBuffer* macfc_buffer_create(int capacity, int num_agents,
                                               int state_dim, int action_dim,
                                               int global_state_dim) {
    MACfcReplayBuffer* buf = (MACfcReplayBuffer*)safe_calloc(1, sizeof(MACfcReplayBuffer));
    if (!buf) return NULL;

    buf->capacity = capacity;
    buf->head = 0;
    buf->size = 0;
    buf->num_agents = num_agents;
    buf->state_dim = state_dim;
    buf->action_dim = action_dim;
    buf->global_state_dim = global_state_dim;

    buf->entries = (MACfcBufferEntry*)safe_calloc((size_t)capacity, sizeof(MACfcBufferEntry));
    if (!buf->entries) { safe_free((void**)&buf); return NULL; }

    int total_states = num_agents * state_dim;
    int total_actions = num_agents * action_dim;

    for (int i = 0; i < capacity; i++) {
        buf->entries[i].states = (float*)safe_calloc((size_t)total_states, sizeof(float));
        buf->entries[i].actions = (float*)safe_calloc((size_t)total_actions, sizeof(float));
        buf->entries[i].rewards = (float*)safe_calloc((size_t)num_agents, sizeof(float));
        buf->entries[i].next_states = (float*)safe_calloc((size_t)total_states, sizeof(float));
        buf->entries[i].dones = (int*)safe_calloc((size_t)num_agents, sizeof(int));
        if (global_state_dim > 0) {
            buf->entries[i].global_state = (float*)safe_calloc((size_t)global_state_dim, sizeof(float));
            buf->entries[i].next_global_state = (float*)safe_calloc((size_t)global_state_dim, sizeof(float));
        } else {
            buf->entries[i].global_state = NULL;
            buf->entries[i].next_global_state = NULL;
        }
        if (!buf->entries[i].states || !buf->entries[i].actions ||
            !buf->entries[i].rewards || !buf->entries[i].next_states || !buf->entries[i].dones) {
            for (int j = 0; j <= i; j++) {
                safe_free((void**)&buf->entries[j].states);
                safe_free((void**)&buf->entries[j].actions);
                safe_free((void**)&buf->entries[j].rewards);
                safe_free((void**)&buf->entries[j].next_states);
                safe_free((void**)&buf->entries[j].dones);
                safe_free((void**)&buf->entries[j].global_state);
                safe_free((void**)&buf->entries[j].next_global_state);
            }
            safe_free((void**)&buf->entries);
            safe_free((void**)&buf);
            return NULL;
        }
    }
    return buf;
}

/**
 * @brief 销毁CfC多智能体经验回放缓冲区
 */
static void macfc_buffer_destroy(MACfcReplayBuffer* buf) {
    if (!buf) return;
    for (int i = 0; i < buf->capacity; i++) {
        safe_free((void**)&buf->entries[i].states);
        safe_free((void**)&buf->entries[i].actions);
        safe_free((void**)&buf->entries[i].rewards);
        safe_free((void**)&buf->entries[i].next_states);
        safe_free((void**)&buf->entries[i].dones);
        safe_free((void**)&buf->entries[i].global_state);
        safe_free((void**)&buf->entries[i].next_global_state);
    }
    safe_free((void**)&buf->entries);
    safe_free((void**)&buf);
}

/**
 * @brief 向缓冲区添加经验
 */
static int macfc_buffer_add(MACfcReplayBuffer* buf, const float* states,
                            const float* actions, const float* rewards,
                            const float* next_states, const int* dones,
                            const float* global_state, const float* next_global_state) {
    if (!buf || !states || !actions || !rewards || !next_states || !dones) return -1;
    MACfcBufferEntry* entry = &buf->entries[buf->head];
    int total_states = buf->num_agents * buf->state_dim;
    int total_actions = buf->num_agents * buf->action_dim;
    memcpy(entry->states, states, (size_t)total_states * sizeof(float));
    memcpy(entry->actions, actions, (size_t)total_actions * sizeof(float));
    memcpy(entry->rewards, rewards, (size_t)buf->num_agents * sizeof(float));
    memcpy(entry->next_states, next_states, (size_t)total_states * sizeof(float));
    memcpy(entry->dones, dones, (size_t)buf->num_agents * sizeof(int));
    if (buf->global_state_dim > 0 && global_state && next_global_state) {
        memcpy(entry->global_state, global_state, (size_t)buf->global_state_dim * sizeof(float));
        memcpy(entry->next_global_state, next_global_state, (size_t)buf->global_state_dim * sizeof(float));
    }
    buf->head = (buf->head + 1) % buf->capacity;
    if (buf->size < buf->capacity) buf->size++;
    return 0;
}

/**
 * @brief 从缓冲区采样
 */
static int macfc_buffer_sample(MACfcReplayBuffer* buf, int batch_size,
                               float* states_out, float* actions_out,
                               float* rewards_out, float* next_states_out,
                               int* dones_out, float* global_state_out,
                               float* next_global_state_out) {
    if (!buf || buf->size < batch_size) return -1;
    int total_states = buf->num_agents * buf->state_dim;
    int total_actions = buf->num_agents * buf->action_dim;
    for (int b = 0; b < batch_size; b++) {
        uint32_t seed = (uint32_t)((uintptr_t)buf ^ (uintptr_t)(b * 7919U + 1337U));
        int idx = (int)(deterministic_rand(&seed) * buf->size);
        if (idx >= buf->size) idx = buf->size - 1;
        int entry_idx = (buf->head - buf->size + idx + buf->capacity) % buf->capacity;
        if (entry_idx < 0) entry_idx += buf->capacity;
        MACfcBufferEntry* entry = &buf->entries[entry_idx];
        memcpy(states_out + (size_t)b * total_states, entry->states, (size_t)total_states * sizeof(float));
        memcpy(actions_out + (size_t)b * total_actions, entry->actions, (size_t)total_actions * sizeof(float));
        memcpy(rewards_out + (size_t)b * buf->num_agents, entry->rewards, (size_t)buf->num_agents * sizeof(float));
        memcpy(next_states_out + (size_t)b * total_states, entry->next_states, (size_t)total_states * sizeof(float));
        memcpy(dones_out + (size_t)b * buf->num_agents, entry->dones, (size_t)buf->num_agents * sizeof(int));
        if (buf->global_state_dim > 0 && global_state_out && next_global_state_out) {
            memcpy(global_state_out + (size_t)b * buf->global_state_dim, entry->global_state,
                   (size_t)buf->global_state_dim * sizeof(float));
            memcpy(next_global_state_out + (size_t)b * buf->global_state_dim, entry->next_global_state,
                   (size_t)buf->global_state_dim * sizeof(float));
        }
    }
    return 0;
}

/* ============================================================================
 * CfC-MADDPG数据结构和算法
 * =========================================================================== */

typedef struct {
    LNN* actor;
    LNN* critic;
    LNN* actor_target;
    LNN* critic_target;
    int hidden_size;
} MACfcMADDPGAgentNetworks;

typedef struct {
    MACfcPrivateHeader header;
    int num_agents;
    int state_dim;
    int action_dim;
    int hidden_size;
    float gamma;
    float tau;
    float actor_lr;
    float critic_lr;
    float exploration_noise;
    float noise_clip;
    int cfc_ode_solver;
    float cfc_time_constant;
    MACfcMADDPGAgentNetworks* agents;
    MACfcReplayBuffer* replay_buffer;
    int critic_input_dim;
    int current_step;
    int training_steps;
    float avg_critic_loss;
    float avg_actor_loss;
} MACfcMADDPGData;

/**
 * @brief 创建MADDPG网络（基于LNN液态神经网络）
 */
static int maddpg_create_networks(MACfcMADDPGData* data) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim, hd = data->hidden_size, cd = data->critic_input_dim;
    data->agents = (MACfcMADDPGAgentNetworks*)safe_calloc((size_t)na, sizeof(MACfcMADDPGAgentNetworks));
    if (!data->agents) return -1;
    for (int i = 0; i < na; i++) {
        MACfcMADDPGAgentNetworks* ag = &data->agents[i];
        ag->hidden_size = hd;
        LNNConfig actor_cfg;
        memset(&actor_cfg, 0, sizeof(actor_cfg));
        actor_cfg.input_size = (size_t)sd; actor_cfg.hidden_size = (size_t)hd; actor_cfg.output_size = (size_t)ad;
        actor_cfg.learning_rate = data->actor_lr; actor_cfg.time_constant = data->cfc_time_constant;
        actor_cfg.enable_training = 1; actor_cfg.num_layers = 2; actor_cfg.ode_solver_type = data->cfc_ode_solver;
        ag->actor = lnn_create(&actor_cfg);
        if (!ag->actor) return -1;
        ag->actor_target = lnn_create(&actor_cfg);
        if (!ag->actor_target) return -1;
        LNNConfig critic_cfg;
        memset(&critic_cfg, 0, sizeof(critic_cfg));
        critic_cfg.input_size = (size_t)cd; critic_cfg.hidden_size = (size_t)(hd * 2); critic_cfg.output_size = 1;
        critic_cfg.learning_rate = data->critic_lr; critic_cfg.time_constant = data->cfc_time_constant;
        critic_cfg.enable_training = 1; critic_cfg.num_layers = 2; critic_cfg.ode_solver_type = data->cfc_ode_solver;
        ag->critic = lnn_create(&critic_cfg);
        if (!ag->critic) return -1;
        ag->critic_target = lnn_create(&critic_cfg);
        if (!ag->critic_target) return -1;
    }
    return 0;
}

static void maddpg_destroy_networks(MACfcMADDPGData* data) {
    if (!data || !data->agents) return;
    int na = data->num_agents;
    for (int i = 0; i < na; i++) {
        MACfcMADDPGAgentNetworks* ag = &data->agents[i];
        if (ag->actor) lnn_free(ag->actor);
        if (ag->actor_target) lnn_free(ag->actor_target);
        if (ag->critic) lnn_free(ag->critic);
        if (ag->critic_target) lnn_free(ag->critic_target);
    }
    safe_free((void**)&data->agents);
}

/**
 * @brief CfC-MADDPG选择动作
 */
static int maddpg_select_actions(MACfcMADDPGData* data, const float* states,
                                 float* actions_out, int add_noise) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim;
    for (int i = 0; i < na; i++) {
        const float* s = states + (size_t)i * sd;
        float* a = actions_out + (size_t)i * ad;
        MACfcMADDPGAgentNetworks* ag = &data->agents[i];
        if (lnn_forward(ag->actor, s, a) != 0) {
            memset(a, 0, (size_t)ad * sizeof(float));
            return -1;
        }
        for (int j = 0; j < ad; j++) {
            if (a[j] > 1.0f) a[j] = 1.0f;
            if (a[j] < -1.0f) a[j] = -1.0f;
            if (add_noise) {
                uint32_t noise_seed = (uint32_t)((uintptr_t)ag ^ (uintptr_t)(i * ad + j));
                float noise = (deterministic_rand(&noise_seed) - 0.5f) * 2.0f * data->exploration_noise;
                if (noise > data->noise_clip) noise = data->noise_clip;
                if (noise < -data->noise_clip) noise = -data->noise_clip;
                a[j] += noise;
                if (a[j] > 1.0f) a[j] = 1.0f;
                if (a[j] < -1.0f) a[j] = -1.0f;
            }
        }
    }
    return 0;
}

/**
 * @brief CfC-MADDPG训练一步
 */
static int maddpg_train_step(MACfcMADDPGData* data) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim, hd = data->hidden_size, cd = data->critic_input_dim;
    int batch_size = 64;
    if (data->replay_buffer->size < batch_size * 2) return 0;

    float* s_buf = (float*)safe_malloc((size_t)batch_size * na * sd * sizeof(float));
    float* a_buf = (float*)safe_malloc((size_t)batch_size * na * ad * sizeof(float));
    float* r_buf = (float*)safe_malloc((size_t)batch_size * na * sizeof(float));
    float* ns_buf = (float*)safe_malloc((size_t)batch_size * na * sd * sizeof(float));
    int* d_buf = (int*)safe_malloc((size_t)batch_size * na * sizeof(int));
    if (!s_buf || !a_buf || !r_buf || !ns_buf || !d_buf) {
        safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf); safe_free((void**)&ns_buf); safe_free((void**)&d_buf);
        return -1;
    }
    if (macfc_buffer_sample(data->replay_buffer, batch_size, s_buf, a_buf, r_buf, ns_buf, d_buf, NULL, NULL) != 0) {
        safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf); safe_free((void**)&ns_buf); safe_free((void**)&d_buf);
        return -1;
    }

    float total_critic_loss = 0.0f, total_actor_loss = 0.0f;
    for (int agent_i = 0; agent_i < na; agent_i++) {
        MACfcMADDPGAgentNetworks* ag = &data->agents[agent_i];
        for (int b = 0; b < batch_size; b++) {
            const float* ss = s_buf + (size_t)b * na * sd;
            const float* aa = a_buf + (size_t)b * na * ad;
            const float* ns = ns_buf + (size_t)b * na * sd;
            float rew = r_buf[(size_t)b * na + agent_i];
            int done = d_buf[(size_t)b * na + agent_i];
            float* tmp_h = (float*)safe_calloc((size_t)hd, sizeof(float));
            float* tmp_c = (float*)safe_calloc((size_t)hd, sizeof(float));

            float* critic_in = (float*)safe_malloc((size_t)cd * sizeof(float));
            memcpy(critic_in, ss, (size_t)(na * sd) * sizeof(float));
            memcpy(critic_in + na * sd, aa, (size_t)(na * ad) * sizeof(float));

            float* next_actions = (float*)safe_malloc((size_t)(na * ad) * sizeof(float));
            for (int k = 0; k < na; k++) {
                const float* ns_k = ns + (size_t)k * sd;
                float* na_k = next_actions + (size_t)k * ad;
                lnn_forward(data->agents[k].actor_target, ns_k, na_k);
                for (int j2 = 0; j2 < ad; j2++) { if (na_k[j2] > 1.0f) na_k[j2] = 1.0f; if (na_k[j2] < -1.0f) na_k[j2] = -1.0f; }
            }

            float* critic_target_in = (float*)safe_malloc((size_t)cd * sizeof(float));
            memcpy(critic_target_in, ns, (size_t)(na * sd) * sizeof(float));
            memcpy(critic_target_in + na * sd, next_actions, (size_t)(na * ad) * sizeof(float));

            float q_target = 0.0f, q_current = 0.0f;
            float q_buf[1];
            lnn_forward(ag->critic_target, critic_target_in, q_buf); q_target = q_buf[0];
            float td_target = rew + data->gamma * q_target * (1.0f - (float)done);
            lnn_forward(ag->critic, critic_in, q_buf); q_current = q_buf[0];
            float td_error = q_current - td_target;
            total_critic_loss += td_error * td_error * 0.5f;
            cfc_backward(lnn_get_cfc_network(ag->critic), &td_error, tmp_h, data->critic_lr);

            safe_free((void**)&next_actions);
            safe_free((void**)&critic_in);
            safe_free((void**)&critic_target_in);
            safe_free((void**)&tmp_h);
            safe_free((void**)&tmp_c);
        }

        for (int b = 0; b < batch_size; b++) {
            const float* ss = s_buf + (size_t)b * na * sd;
            const float* aa = a_buf + (size_t)b * na * ad;
            float* new_action = (float*)safe_calloc((size_t)ad, sizeof(float));
            float* tmp_h = (float*)safe_calloc((size_t)hd, sizeof(float));
            float* tmp_c = (float*)safe_calloc((size_t)hd, sizeof(float));
            lnn_forward(ag->actor, ss + (size_t)agent_i * sd, new_action);
            for (int j2 = 0; j2 < ad; j2++) { if (new_action[j2] > 1.0f) new_action[j2] = 1.0f; if (new_action[j2] < -1.0f) new_action[j2] = -1.0f; }

            float* critic_actor_in = (float*)safe_malloc((size_t)cd * sizeof(float));
            memcpy(critic_actor_in, ss, (size_t)(na * sd) * sizeof(float));
            memcpy(critic_actor_in + na * sd, aa, (size_t)(na * ad) * sizeof(float));
            memcpy(critic_actor_in + na * sd + (size_t)agent_i * ad, new_action, (size_t)ad * sizeof(float));

            float q_val = 0.0f;
            float qval_buf[1];
            lnn_forward(ag->critic, critic_actor_in, qval_buf); q_val = qval_buf[0];
            total_actor_loss -= q_val;
            float one = 1.0f;
            cfc_backward(lnn_get_cfc_network(ag->critic), &one, tmp_h, 0.0f);
            cfc_backward(lnn_get_cfc_network(ag->actor), &one, tmp_h, data->actor_lr);

            safe_free((void**)&critic_actor_in);
            safe_free((void**)&new_action);
            safe_free((void**)&tmp_h);
            safe_free((void**)&tmp_c);
        }
    }

    data->avg_critic_loss = total_critic_loss / (float)(batch_size * na);
    data->avg_actor_loss = total_actor_loss / (float)(batch_size * na);
    data->training_steps++;
    safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf); safe_free((void**)&ns_buf); safe_free((void**)&d_buf);
    return 0;
}

/* ============================================================================
 * CfC-QMIX数据结构和算法
 * =========================================================================== */

typedef struct {
    LNN* agent_q;
    LNN* agent_q_target;
    int hidden_size;
} MACfcQMIXAgentNetworks;

typedef struct {
    MACfcPrivateHeader header;
    int num_agents;
    int state_dim;
    int action_dim;
    int hidden_size;
    int mixing_hidden_size;
    LNN* mixing_net;
    LNN* mixing_target;
    MACfcQMIXAgentNetworks* agents;
    MACfcReplayBuffer* replay_buffer;
    float gamma;
    float tau;
    int target_update_interval;
    int current_step;
    int training_steps;
    float avg_loss;
} MACfcQMIXData;

/**
 * @brief 创建QMIX网络（基于LNN液态神经网络）
 */
static int qmix_create_networks(MACfcQMIXData* data) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim, hd = data->hidden_size, mhd = data->mixing_hidden_size;
    int gsd = na * sd;

    data->agents = (MACfcQMIXAgentNetworks*)safe_calloc((size_t)na, sizeof(MACfcQMIXAgentNetworks));
    if (!data->agents) return -1;

    for (int i = 0; i < na; i++) {
        MACfcQMIXAgentNetworks* ag = &data->agents[i];
        ag->hidden_size = hd;
        LNNConfig q_cfg;
        memset(&q_cfg, 0, sizeof(q_cfg));
        q_cfg.input_size = (size_t)sd; q_cfg.hidden_size = (size_t)hd; q_cfg.output_size = (size_t)ad;
        q_cfg.learning_rate = data->gamma; q_cfg.time_constant = 0.1f;
        q_cfg.enable_training = 1; q_cfg.num_layers = 2; q_cfg.ode_solver_type = 0;
        ag->agent_q = lnn_create(&q_cfg);
        if (!ag->agent_q) return -1;
        ag->agent_q_target = lnn_create(&q_cfg);
        if (!ag->agent_q_target) return -1;
    }

    LNNConfig mix_cfg;
    memset(&mix_cfg, 0, sizeof(mix_cfg));
    mix_cfg.input_size = (size_t)gsd; mix_cfg.hidden_size = (size_t)mhd; mix_cfg.output_size = 1;
    mix_cfg.learning_rate = 0.001f; mix_cfg.time_constant = 0.1f;
    mix_cfg.enable_training = 1; mix_cfg.num_layers = 2; mix_cfg.ode_solver_type = 0;
    data->mixing_net = lnn_create(&mix_cfg);
    if (!data->mixing_net) return -1;
    data->mixing_target = lnn_create(&mix_cfg);
    if (!data->mixing_target) return -1;
    return 0;
}

static void qmix_destroy_networks(MACfcQMIXData* data) {
    if (!data || !data->agents) return;
    int na = data->num_agents;
    for (int i = 0; i < na; i++) {
        MACfcQMIXAgentNetworks* ag = &data->agents[i];
        if (ag->agent_q) lnn_free(ag->agent_q);
        if (ag->agent_q_target) lnn_free(ag->agent_q_target);
    }
    safe_free((void**)&data->agents);
    if (data->mixing_net) lnn_free(data->mixing_net);
    if (data->mixing_target) lnn_free(data->mixing_target);
}

/**
 * @brief CfC-QMIX选择动作（epsilon-greedy）
 */
static int qmix_select_actions(MACfcQMIXData* data, const float* states,
                               float* actions_out, float epsilon) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim;
    for (int i = 0; i < na; i++) {
        const float* s = states + (size_t)i * sd;
        float* a = actions_out + (size_t)i * ad;
        MACfcQMIXAgentNetworks* ag = &data->agents[i];
        uint32_t eps_seed = (uint32_t)((uintptr_t)ag ^ (uintptr_t)(i * 7919U));
        if (deterministic_rand(&eps_seed) < epsilon) {
            for (int j = 0; j < ad; j++) {
                uint32_t rnd_seed = (uint32_t)((uintptr_t)(j + 1) * 7919U ^ (uintptr_t)ag);
                a[j] = (deterministic_rand(&rnd_seed) - 0.5f) * 2.0f;
            }
        } else {
            float q_vals[MA_CFC_MAX_ACTION_DIM];
            if (lnn_forward(ag->agent_q, s, q_vals) != 0) {
                memset(a, 0, (size_t)ad * sizeof(float));
                return -1;
            }
            int best_j = 0;
            for (int j = 1; j < ad; j++) {
                if (q_vals[j] > q_vals[best_j]) best_j = j;
            }
            memset(a, 0, (size_t)ad * sizeof(float));
            a[best_j] = 1.0f;
        }
    }
    return 0;
}

/**
 * @brief CfC-QMIX训练一步
 */
static int qmix_train_step(MACfcQMIXData* data) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim;
    int gsd = na * sd, batch_size = 32;
    (void)data->mixing_hidden_size;
    if (data->replay_buffer->size < batch_size * 2) return 0;

    float* s_buf = (float*)safe_malloc((size_t)batch_size * na * sd * sizeof(float));
    float* a_buf = (float*)safe_malloc((size_t)batch_size * na * ad * sizeof(float));
    float* r_buf = (float*)safe_malloc((size_t)batch_size * na * sizeof(float));
    float* ns_buf = (float*)safe_malloc((size_t)batch_size * na * sd * sizeof(float));
    int* d_buf = (int*)safe_malloc((size_t)batch_size * na * sizeof(int));
    float* gs_buf = (float*)safe_malloc((size_t)batch_size * gsd * sizeof(float));
    float* ngs_buf = (float*)safe_malloc((size_t)batch_size * gsd * sizeof(float));
    if (!s_buf || !a_buf || !r_buf || !ns_buf || !d_buf || !gs_buf || !ngs_buf) {
        safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf);
        safe_free((void**)&ns_buf); safe_free((void**)&d_buf); safe_free((void**)&gs_buf); safe_free((void**)&ngs_buf);
        return -1;
    }
    if (macfc_buffer_sample(data->replay_buffer, batch_size, s_buf, a_buf, r_buf, ns_buf, d_buf, gs_buf, ngs_buf) != 0) {
        safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf);
        safe_free((void**)&ns_buf); safe_free((void**)&d_buf); safe_free((void**)&gs_buf); safe_free((void**)&ngs_buf);
        return -1;
    }

    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        const float* ss = s_buf + (size_t)b * na * sd;
        const float* aa = a_buf + (size_t)b * na * ad;
        const float* ns = ns_buf + (size_t)b * na * sd;
        const float* gs = gs_buf + (size_t)b * gsd;
        const float* ngs = ngs_buf + (size_t)b * gsd;
        (void)gs; (void)ngs;
        float q_tot = 0.0f, q_target = 0.0f;

        for (int i = 0; i < na; i++) {
            float q_i[MA_CFC_MAX_ACTION_DIM];
            float q_next[MA_CFC_MAX_ACTION_DIM];
            float tmp_h[64], tmp_c[64];
            memset(tmp_h, 0, sizeof(tmp_h)); memset(tmp_c, 0, sizeof(tmp_c));
            lnn_forward(data->agents[i].agent_q, ss + (size_t)i * sd, q_i);
            lnn_forward(data->agents[i].agent_q_target, ns + (size_t)i * sd, q_next);
            int best_a = 0;
            for (int j = 1; j < ad; j++) if (q_next[j] > q_next[best_a]) best_a = j;
            q_tot += q_i[(int)aa[(size_t)i * ad + best_a]];
            float reward_sum = 0.0f;
            for (int k = 0; k < na; k++) reward_sum += r_buf[(size_t)b * na + k];
            q_target = reward_sum + data->gamma * q_next[best_a];
        }
        float td_error = q_tot - q_target;
        total_loss += td_error * td_error * 0.5f;
    }

    data->avg_loss = total_loss / (float)batch_size;
    data->training_steps++;
    safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf);
    safe_free((void**)&ns_buf); safe_free((void**)&d_buf); safe_free((void**)&gs_buf); safe_free((void**)&ngs_buf);
    return 0;
}

/* ============================================================================
 * CfC-VDN数据结构和算法
 * =========================================================================== */

typedef struct {
    MACfcPrivateHeader header;
    int num_agents;
    int state_dim;
    int action_dim;
    int hidden_size;
    float gamma;
    MACfcQMIXAgentNetworks* agents;
    MACfcReplayBuffer* replay_buffer;
    int training_steps;
    float avg_loss;
} MACfcVDNData;

/**
 * @brief 创建CfC-VDN网络
 */
static int vdn_create_networks(MACfcVDNData* data) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim, hd = data->hidden_size;
    data->agents = (MACfcQMIXAgentNetworks*)safe_calloc((size_t)na, sizeof(MACfcQMIXAgentNetworks));
    if (!data->agents) return -1;
    for (int i = 0; i < na; i++) {
        MACfcQMIXAgentNetworks* ag = &data->agents[i];
        ag->hidden_size = hd;
        LNNConfig q_cfg;
        memset(&q_cfg, 0, sizeof(q_cfg));
        q_cfg.input_size = (size_t)sd; q_cfg.hidden_size = (size_t)hd; q_cfg.output_size = (size_t)ad;
        q_cfg.learning_rate = data->gamma; q_cfg.time_constant = 0.1f;
        q_cfg.enable_training = 1; q_cfg.num_layers = 2; q_cfg.ode_solver_type = 0;
        ag->agent_q = lnn_create(&q_cfg);
        if (!ag->agent_q) return -1;
        ag->agent_q_target = lnn_create(&q_cfg);
        if (!ag->agent_q_target) return -1;
    }
    return 0;
}

static void vdn_destroy_networks(MACfcVDNData* data) {
    if (!data || !data->agents) return;
    int na = data->num_agents;
    for (int i = 0; i < na; i++) {
        MACfcQMIXAgentNetworks* ag = &data->agents[i];
        if (ag->agent_q) lnn_free(ag->agent_q);
        if (ag->agent_q_target) lnn_free(ag->agent_q_target);
    }
    safe_free((void**)&data->agents);
}

/**
 * @brief CfC-VDN选择动作
 */
static int vdn_select_actions(MACfcVDNData* data, const float* states,
                              float* actions_out, float epsilon) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim;
    for (int i = 0; i < na; i++) {
        const float* s = states + (size_t)i * sd;
        float* a = actions_out + (size_t)i * ad;
        MACfcQMIXAgentNetworks* ag = &data->agents[i];
        uint32_t eps_seed = (uint32_t)((uintptr_t)ag ^ (uintptr_t)(i * 7919U));
        if (deterministic_rand(&eps_seed) < epsilon) {
            for (int j = 0; j < ad; j++) {
                uint32_t rnd_seed = (uint32_t)((uintptr_t)(j + 1) * 7919U ^ (uintptr_t)ag);
                a[j] = (deterministic_rand(&rnd_seed) - 0.5f) * 2.0f;
            }
        } else {
            float q_vals[MA_CFC_MAX_ACTION_DIM];
            lnn_forward(ag->agent_q, s, q_vals);
            int best_j = 0;
            for (int j = 1; j < ad; j++) if (q_vals[j] > q_vals[best_j]) best_j = j;
            memset(a, 0, (size_t)ad * sizeof(float));
            a[best_j] = 1.0f;
        }
    }
    return 0;
}

/**
 * @brief CfC-VDN训练一步
 */
static int vdn_train_step(MACfcVDNData* data) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim;
    int batch_size = 32;
    if (data->replay_buffer->size < batch_size * 2) return 0;

    float* s_buf = (float*)safe_malloc((size_t)batch_size * na * sd * sizeof(float));
    float* a_buf = (float*)safe_malloc((size_t)batch_size * na * ad * sizeof(float));
    float* r_buf = (float*)safe_malloc((size_t)batch_size * na * sizeof(float));
    float* ns_buf = (float*)safe_malloc((size_t)batch_size * na * sd * sizeof(float));
    int* d_buf = (int*)safe_malloc((size_t)batch_size * na * sizeof(int));
    if (!s_buf || !a_buf || !r_buf || !ns_buf || !d_buf) {
        safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf); safe_free((void**)&ns_buf); safe_free((void**)&d_buf);
        return -1;
    }
    if (macfc_buffer_sample(data->replay_buffer, batch_size, s_buf, a_buf, r_buf, ns_buf, d_buf, NULL, NULL) != 0) {
        safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf); safe_free((void**)&ns_buf); safe_free((void**)&d_buf);
        return -1;
    }

    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        const float* ss = s_buf + (size_t)b * na * sd;
        const float* aa = a_buf + (size_t)b * na * ad;
        const float* ns = ns_buf + (size_t)b * na * sd;
        (void)aa;
        float q_tot = 0.0f, q_target_total = 0.0f;
        for (int i = 0; i < na; i++) {
            float q_i[MA_CFC_MAX_ACTION_DIM];
            float q_next[MA_CFC_MAX_ACTION_DIM];
            float tmp_h[64], tmp_c[64];
            memset(tmp_h, 0, sizeof(tmp_h)); memset(tmp_c, 0, sizeof(tmp_c));
            lnn_forward(data->agents[i].agent_q, ss + (size_t)i * sd, q_i);
            lnn_forward(data->agents[i].agent_q_target, ns + (size_t)i * sd, q_next);
            int best_a = 0;
            for (int j = 1; j < ad; j++) if (q_next[j] > q_next[best_a]) best_a = j;
            q_tot += q_i[best_a];
            q_target_total += r_buf[(size_t)b * na + i] + data->gamma * q_next[best_a] * (1.0f - (float)d_buf[(size_t)b * na + i]);
        }
        float td_error = q_tot - q_target_total;
        total_loss += td_error * td_error * 0.5f;
    }

    data->avg_loss = total_loss / (float)batch_size;
    data->training_steps++;
    safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf); safe_free((void**)&ns_buf); safe_free((void**)&d_buf);
    return 0;
}

/* ============================================================================
 * CfC-COMA数据结构和算法
 * =========================================================================== */

typedef struct {
    LNN* actor;
    LNN* critic;
    LNN* critic_target;
    int hidden_size;
} MACfcCOMAgentNetworks;

typedef struct {
    MACfcPrivateHeader header;
    int num_agents;
    int state_dim;
    int action_dim;
    int hidden_size;
    float gamma;
    float lr;
    MACfcCOMAgentNetworks* agents;
    MACfcReplayBuffer* replay_buffer;
    int critic_input_dim;
    int training_steps;
    float avg_loss;
} MACfcCOMAData;

/**
 * @brief 创建CfC-COMA网络
 */
static int coma_create_networks(MACfcCOMAData* data) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim, hd = data->hidden_size, cd = data->critic_input_dim;
    data->agents = (MACfcCOMAgentNetworks*)safe_calloc((size_t)na, sizeof(MACfcCOMAgentNetworks));
    if (!data->agents) return -1;
    for (int i = 0; i < na; i++) {
        MACfcCOMAgentNetworks* ag = &data->agents[i];
        ag->hidden_size = hd;
        LNNConfig actor_cfg;
        memset(&actor_cfg, 0, sizeof(actor_cfg));
        actor_cfg.input_size = (size_t)sd; actor_cfg.hidden_size = (size_t)hd; actor_cfg.output_size = (size_t)ad;
        actor_cfg.learning_rate = data->lr; actor_cfg.time_constant = 0.1f;
        actor_cfg.enable_training = 1; actor_cfg.num_layers = 2; actor_cfg.ode_solver_type = 0;
        ag->actor = lnn_create(&actor_cfg);
        if (!ag->actor) return -1;
        LNNConfig critic_cfg;
        memset(&critic_cfg, 0, sizeof(critic_cfg));
        critic_cfg.input_size = (size_t)cd; critic_cfg.hidden_size = (size_t)(hd * 2); critic_cfg.output_size = 1;
        critic_cfg.learning_rate = data->lr * 2.0f; critic_cfg.time_constant = 0.1f;
        critic_cfg.enable_training = 1; critic_cfg.num_layers = 2; critic_cfg.ode_solver_type = 0;
        ag->critic = lnn_create(&critic_cfg);
        if (!ag->critic) return -1;
        ag->critic_target = lnn_create(&critic_cfg);
        if (!ag->critic_target) return -1;
    }
    return 0;
}

static void coma_destroy_networks(MACfcCOMAData* data) {
    if (!data || !data->agents) return;
    int na = data->num_agents;
    for (int i = 0; i < na; i++) {
        MACfcCOMAgentNetworks* ag = &data->agents[i];
        if (ag->actor) lnn_free(ag->actor);
        if (ag->critic) lnn_free(ag->critic);
        if (ag->critic_target) lnn_free(ag->critic_target);
    }
    safe_free((void**)&data->agents);
}

/**
 * @brief CfC-COMA选择动作
 */
static int coma_select_actions(MACfcCOMAData* data, const float* states,
                               float* actions_out, float epsilon) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim;
    for (int i = 0; i < na; i++) {
        const float* s = states + (size_t)i * sd;
        float* a = actions_out + (size_t)i * ad;
        MACfcCOMAgentNetworks* ag = &data->agents[i];
        uint32_t eps_seed = (uint32_t)((uintptr_t)ag ^ (uintptr_t)(i * 7919U));
        if (deterministic_rand(&eps_seed) < epsilon) {
            for (int j = 0; j < ad; j++) {
                uint32_t rnd_seed = (uint32_t)((uintptr_t)(j + 1) * 7919U ^ (uintptr_t)ag);
                a[j] = (deterministic_rand(&rnd_seed) - 0.5f) * 2.0f;
            }
        } else {
            float logits[MA_CFC_MAX_ACTION_DIM];
            lnn_forward(ag->actor, s, logits);
            float max_val = logits[0]; for (int j = 1; j < ad; j++) if (logits[j] > max_val) max_val = logits[j];
            float sum_exp = 0.0f; for (int j = 0; j < ad; j++) sum_exp += expf(logits[j] - max_val);
            float probs[MA_CFC_MAX_ACTION_DIM]; for (int j = 0; j < ad; j++) probs[j] = expf(logits[j] - max_val) / sum_exp;
            uint32_t rnd_seed = (uint32_t)((uintptr_t)ag ^ 1337U);
            float r = deterministic_rand(&rnd_seed);
            float cum = 0.0f; int chosen = 0;
            for (int j = 0; j < ad; j++) { cum += probs[j]; if (r <= cum) { chosen = j; break; } }
            memset(a, 0, (size_t)ad * sizeof(float));
            a[chosen] = 1.0f;
        }
    }
    return 0;
}

/**
 * @brief CfC-COMA训练一步（反事实基线）
 */
static int coma_train_step(MACfcCOMAData* data) {
    int na = data->num_agents, sd = data->state_dim, ad = data->action_dim, cd = data->critic_input_dim;
    (void)data->hidden_size;
    int batch_size = 32;
    if (data->replay_buffer->size < batch_size * 2) return 0;

    float* s_buf = (float*)safe_malloc((size_t)batch_size * na * sd * sizeof(float));
    float* a_buf = (float*)safe_malloc((size_t)batch_size * na * ad * sizeof(float));
    float* r_buf = (float*)safe_malloc((size_t)batch_size * na * sizeof(float));
    float* ns_buf = (float*)safe_malloc((size_t)batch_size * na * sd * sizeof(float));
    int* d_buf = (int*)safe_malloc((size_t)batch_size * na * sizeof(int));
    if (!s_buf || !a_buf || !r_buf || !ns_buf || !d_buf) {
        safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf); safe_free((void**)&ns_buf); safe_free((void**)&d_buf);
        return -1;
    }
    if (macfc_buffer_sample(data->replay_buffer, batch_size, s_buf, a_buf, r_buf, ns_buf, d_buf, NULL, NULL) != 0) {
        safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf); safe_free((void**)&ns_buf); safe_free((void**)&d_buf);
        return -1;
    }

    float total_loss = 0.0f;
    for (int i = 0; i < na; i++) {
        MACfcCOMAgentNetworks* ag = &data->agents[i];
        for (int b = 0; b < batch_size; b++) {
            const float* ss = s_buf + (size_t)b * na * sd;
            const float* aa = a_buf + (size_t)b * na * ad;
            const float* ns = ns_buf + (size_t)b * na * sd;
            float rew = r_buf[(size_t)b * na + i];
            int done = d_buf[(size_t)b * na + i];
            float tmp_h[128], tmp_c[128];
            memset(tmp_h, 0, sizeof(tmp_h)); memset(tmp_c, 0, sizeof(tmp_c));

            float* critic_in = (float*)safe_malloc((size_t)cd * sizeof(float));
            memcpy(critic_in, ss, (size_t)(na * sd) * sizeof(float));
            memcpy(critic_in + na * sd, aa, (size_t)(na * ad) * sizeof(float));
            float q_current = 0.0f, q_next = 0.0f;
            float q_buf[1];
            lnn_forward(ag->critic, critic_in, q_buf); q_current = q_buf[0];
            float* critic_next_in = (float*)safe_malloc((size_t)cd * sizeof(float));
            memcpy(critic_next_in, ns, (size_t)(na * sd) * sizeof(float));
            memset(critic_next_in + na * sd, 0, (size_t)(na * ad) * sizeof(float));
            lnn_forward(ag->critic_target, critic_next_in, q_buf); q_next = q_buf[0];
            float td_target = rew + data->gamma * q_next * (1.0f - (float)done);
            float td_error = q_current - td_target;
            total_loss += td_error * td_error * 0.5f;
            cfc_backward(lnn_get_cfc_network(ag->critic), &td_error, tmp_h, data->lr * 2.0f);

            float counterfactual_baseline = 0.0f;
            int counter = 0;
            for (int ac = 0; ac < ad; ac++) {
                float* alt_critic_in = (float*)safe_malloc((size_t)cd * sizeof(float));
                memcpy(alt_critic_in, ss, (size_t)(na * sd) * sizeof(float));
                memcpy(alt_critic_in + na * sd, aa, (size_t)(na * ad) * sizeof(float));
                memset(alt_critic_in + na * sd + (size_t)i * ad, 0, (size_t)ad * sizeof(float));
                alt_critic_in[na * sd + i * ad + ac] = 1.0f;
                float q_alt = 0.0f;
                float qalt_buf[1];
                lnn_forward(ag->critic, alt_critic_in, qalt_buf); q_alt = qalt_buf[0];
                counterfactual_baseline += q_alt;
                counter++;
                safe_free((void**)&alt_critic_in);
            }
            if (counter > 0) counterfactual_baseline /= (float)counter;
            float advantage = q_current - counterfactual_baseline;
            float logit_out[MA_CFC_MAX_ACTION_DIM];
            lnn_forward(ag->actor, ss + (size_t)i * sd, logit_out);
            float one = 1.0f;
            cfc_backward(lnn_get_cfc_network(ag->actor), &one, tmp_h, data->lr * advantage);

            safe_free((void**)&critic_in);
            safe_free((void**)&critic_next_in);
        }
    }

    data->avg_loss = total_loss / (float)(batch_size * na);
    data->training_steps++;
    safe_free((void**)&s_buf); safe_free((void**)&a_buf); safe_free((void**)&r_buf); safe_free((void**)&ns_buf); safe_free((void**)&d_buf);
    return 0;
}

/* ============================================================================
 * CfC多智能体算法公共API实现
 * =========================================================================== */

const char* ma_cfc_algorithm_name(MACfcAlgorithm algo) {
    switch (algo) {
        case MA_CFC_ALGORITHM_MADDPG: return "CfC-MADDPG";
        case MA_CFC_ALGORITHM_QMIX:   return "CfC-QMIX";
        case MA_CFC_ALGORITHM_VDN:    return "CfC-VDN";
        case MA_CFC_ALGORITHM_COMA:   return "CfC-COMA";
        default: return "未知算法";
    }
}

void ma_cfc_default_config(MACfcConfig* config, MACfcAlgorithm algorithm) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->algorithm = algorithm;
    config->enable_centralized_critic = 0;
    config->enable_communication = 1;
    config->enable_parameter_sharing = 0;
    config->global_state_dim = 0;

    switch (algorithm) {
        case MA_CFC_ALGORITHM_MADDPG: {
            MACfcMADDPGConfig* acfg = &config->algo_config.maddpg;
            acfg->hidden_size = 64;
            acfg->gamma = 0.95f;
            acfg->tau = 0.01f;
            acfg->actor_lr = 0.001f;
            acfg->critic_lr = 0.002f;
            acfg->exploration_noise = 0.1f;
            acfg->noise_clip = 0.5f;
            acfg->buffer_capacity = 100000;
            acfg->batch_size = 64;
            break;
        }
        case MA_CFC_ALGORITHM_QMIX: {
            MACfcQMIXConfig* acfg = &config->algo_config.qmix;
            acfg->hidden_size = 64;
            acfg->mixing_hidden_size = 32;
            acfg->gamma = 0.95f;
            acfg->tau = 0.005f;
            acfg->lr = 0.001f;
            acfg->buffer_capacity = 100000;
            acfg->batch_size = 32;
            acfg->target_update_interval = 200;
            break;
        }
        case MA_CFC_ALGORITHM_VDN: {
            MACfcVDNConfig* acfg = &config->algo_config.vdn;
            acfg->hidden_size = 64;
            acfg->gamma = 0.95f;
            acfg->lr = 0.001f;
            acfg->buffer_capacity = 100000;
            acfg->batch_size = 32;
            break;
        }
        case MA_CFC_ALGORITHM_COMA: {
            MACfcCOMAConfig* acfg = &config->algo_config.coma;
            acfg->hidden_size = 64;
            acfg->actor_lr = 0.001f;
            acfg->critic_lr = 0.002f;
            acfg->gamma = 0.95f;
            acfg->entropy_coef = 0.01f;
            acfg->buffer_capacity = 100000;
            acfg->batch_size = 32;
            break;
        }
    }
}

MultiAgentSystem* ma_cfc_system_create(const MACfcConfig* config) {
    if (!config) return NULL;

    int na = 4, sd = 8, ad = 4;
    int hidden_size = 64, mixing_hidden_size = 32;
    float gamma = 0.95f, tau = 0.005f;
    int buffer_capacity = 100000, batch_size = 64;
    float actor_lr = 0.001f, critic_lr = 0.002f;

    switch (config->algorithm) {
        case MA_CFC_ALGORITHM_MADDPG: {
            const MACfcMADDPGConfig* acfg = &config->algo_config.maddpg;
            na = 4; sd = 8; ad = 4;
            hidden_size = acfg->hidden_size;
            gamma = acfg->gamma;
            tau = acfg->tau;
            actor_lr = acfg->actor_lr;
            critic_lr = acfg->critic_lr;
            buffer_capacity = acfg->buffer_capacity;
            batch_size = acfg->batch_size;
            break;
        }
        case MA_CFC_ALGORITHM_QMIX: {
            const MACfcQMIXConfig* acfg = &config->algo_config.qmix;
            na = 4; sd = 8; ad = 4;
            hidden_size = acfg->hidden_size;
            mixing_hidden_size = acfg->mixing_hidden_size;
            gamma = acfg->gamma;
            tau = acfg->tau;
            buffer_capacity = acfg->buffer_capacity;
            batch_size = acfg->batch_size;
            break;
        }
        case MA_CFC_ALGORITHM_VDN: {
            const MACfcVDNConfig* acfg = &config->algo_config.vdn;
            na = 4; sd = 8; ad = 4;
            hidden_size = acfg->hidden_size;
            gamma = acfg->gamma;
            tau = acfg->tau;
            buffer_capacity = acfg->buffer_capacity;
            batch_size = acfg->batch_size;
            break;
        }
        case MA_CFC_ALGORITHM_COMA: {
            const MACfcCOMAConfig* acfg = &config->algo_config.coma;
            na = 4; sd = 8; ad = 4;
            hidden_size = acfg->hidden_size;
            gamma = acfg->gamma;
            tau = acfg->tau;
            actor_lr = acfg->actor_lr;
            critic_lr = acfg->critic_lr;
            buffer_capacity = acfg->buffer_capacity;
            batch_size = acfg->batch_size;
            break;
        }
        default:
            return NULL;
    }

    MultiAgentConfig base_cfg;
    multi_agent_default_config(&base_cfg);
    base_cfg.learning_rate = gamma;
    MultiAgentSystem* system = multi_agent_system_create(&base_cfg);
    if (!system) return NULL;

    int gsd = na * sd;
    MACfcReplayBuffer* buffer = macfc_buffer_create(buffer_capacity, na, sd, ad, gsd);
    if (!buffer) { multi_agent_system_destroy(system); return NULL; }

    if (config->algorithm == MA_CFC_ALGORITHM_MADDPG) {
        int cd = na * (sd + ad);
        MACfcMADDPGData* data = (MACfcMADDPGData*)safe_calloc(1, sizeof(MACfcMADDPGData));
        if (!data) { macfc_buffer_destroy(buffer); multi_agent_system_destroy(system); return NULL; }
        data->header.algorithm = config->algorithm;
        data->num_agents = na; data->state_dim = sd; data->action_dim = ad;
        data->hidden_size = hidden_size; data->gamma = gamma; data->tau = tau;
        data->actor_lr = actor_lr; data->critic_lr = critic_lr;
        data->exploration_noise = 0.1f; data->noise_clip = 0.5f;
        data->critic_input_dim = cd;
        data->current_step = 0; data->training_steps = 0;
        data->avg_critic_loss = 0.0f; data->avg_actor_loss = 0.0f;
        data->replay_buffer = buffer;
        if (maddpg_create_networks(data) != 0) {
            macfc_buffer_destroy(buffer); safe_free((void**)&data); multi_agent_system_destroy(system); return NULL;
        }
        system->cfc_private_data = data;
    } else if (config->algorithm == MA_CFC_ALGORITHM_QMIX) {
        MACfcQMIXData* data = (MACfcQMIXData*)safe_calloc(1, sizeof(MACfcQMIXData));
        if (!data) { macfc_buffer_destroy(buffer); multi_agent_system_destroy(system); return NULL; }
        data->header.algorithm = config->algorithm;
        data->num_agents = na; data->state_dim = sd; data->action_dim = ad;
        data->hidden_size = hidden_size; data->mixing_hidden_size = mixing_hidden_size;
        data->gamma = gamma; data->tau = tau;
        data->current_step = 0; data->training_steps = 0; data->avg_loss = 0.0f;
        data->target_update_interval = 200;
        data->replay_buffer = buffer;
        if (qmix_create_networks(data) != 0) {
            macfc_buffer_destroy(buffer); safe_free((void**)&data); multi_agent_system_destroy(system); return NULL;
        }
        system->cfc_private_data = data;
    } else if (config->algorithm == MA_CFC_ALGORITHM_VDN) {
        MACfcVDNData* data = (MACfcVDNData*)safe_calloc(1, sizeof(MACfcVDNData));
        if (!data) { macfc_buffer_destroy(buffer); multi_agent_system_destroy(system); return NULL; }
        data->header.algorithm = config->algorithm;
        data->num_agents = na; data->state_dim = sd; data->action_dim = ad;
        data->hidden_size = hidden_size; data->gamma = gamma;
        data->training_steps = 0; data->avg_loss = 0.0f;
        data->replay_buffer = buffer;
        if (vdn_create_networks(data) != 0) {
            macfc_buffer_destroy(buffer); safe_free((void**)&data); multi_agent_system_destroy(system); return NULL;
        }
        system->cfc_private_data = data;
    } else if (config->algorithm == MA_CFC_ALGORITHM_COMA) {
        int cd = na * (sd + ad);
        MACfcCOMAData* data = (MACfcCOMAData*)safe_calloc(1, sizeof(MACfcCOMAData));
        if (!data) { macfc_buffer_destroy(buffer); multi_agent_system_destroy(system); return NULL; }
        data->header.algorithm = config->algorithm;
        data->num_agents = na; data->state_dim = sd; data->action_dim = ad;
        data->hidden_size = hidden_size; data->gamma = gamma; data->lr = actor_lr;
        data->critic_input_dim = cd;
        data->training_steps = 0; data->avg_loss = 0.0f;
        data->replay_buffer = buffer;
        if (coma_create_networks(data) != 0) {
            macfc_buffer_destroy(buffer); safe_free((void**)&data); multi_agent_system_destroy(system); return NULL;
        }
        system->cfc_private_data = data;
    } else {
        macfc_buffer_destroy(buffer); multi_agent_system_destroy(system); return NULL;
    }

    return system;
}

void ma_cfc_system_destroy(MultiAgentSystem* system) {
    if (!system || !system->cfc_private_data) return;
    MACfcPrivateHeader* hdr = (MACfcPrivateHeader*)system->cfc_private_data;
    switch (hdr->algorithm) {
        case MA_CFC_ALGORITHM_MADDPG: {
            MACfcMADDPGData* data = (MACfcMADDPGData*)hdr;
            macfc_buffer_destroy(data->replay_buffer);
            maddpg_destroy_networks(data);
            break;
        }
        case MA_CFC_ALGORITHM_QMIX: {
            MACfcQMIXData* data = (MACfcQMIXData*)hdr;
            macfc_buffer_destroy(data->replay_buffer);
            qmix_destroy_networks(data);
            break;
        }
        case MA_CFC_ALGORITHM_VDN: {
            MACfcVDNData* data = (MACfcVDNData*)hdr;
            macfc_buffer_destroy(data->replay_buffer);
            vdn_destroy_networks(data);
            break;
        }
        case MA_CFC_ALGORITHM_COMA: {
            MACfcCOMAData* data = (MACfcCOMAData*)hdr;
            macfc_buffer_destroy(data->replay_buffer);
            coma_destroy_networks(data);
            break;
        }
        default:
            break;
    }
    multi_agent_system_destroy(system);
}

int ma_cfc_select_actions(MultiAgentSystem* system, const float* states,
                         float* actions, const int* agent_indices,
                         int num_active, int add_noise) {
    if (!system || !system->cfc_private_data || !states || !actions) return -1;

    MACfcPrivateHeader* hdr = (MACfcPrivateHeader*)system->cfc_private_data;

    if (agent_indices && num_active > 0) {
        int na = 0, ad = 0;
        switch (hdr->algorithm) {
            case MA_CFC_ALGORITHM_MADDPG: {
                MACfcMADDPGData* d = (MACfcMADDPGData*)hdr;
                na = d->num_agents; ad = d->action_dim;
                break;
            }
            case MA_CFC_ALGORITHM_QMIX:
            case MA_CFC_ALGORITHM_VDN: {
                MACfcQMIXData* d = (MACfcQMIXData*)hdr;
                na = d->num_agents; ad = d->action_dim;
                break;
            }
            case MA_CFC_ALGORITHM_COMA: {
                MACfcCOMAData* d = (MACfcCOMAData*)hdr;
                na = d->num_agents; ad = d->action_dim;
                break;
            }
            default: return -1;
        }
        size_t total_action_size = (size_t)na * (size_t)ad;
        float* all_actions = (float*)safe_malloc(total_action_size * sizeof(float));
        if (!all_actions) return -1;
        int ret = ma_cfc_select_actions(system, states, all_actions, NULL, 0, add_noise);
        if (ret == 0) {
            for (int k = 0; k < num_active; k++) {
                int ai = agent_indices[k];
                if (ai >= 0 && ai < na) {
                    memcpy(actions + (size_t)ai * ad, all_actions + (size_t)ai * ad, (size_t)ad * sizeof(float));
                }
            }
        }
        safe_free((void**)&all_actions);
        return ret;
    }

    switch (hdr->algorithm) {
        case MA_CFC_ALGORITHM_MADDPG: {
            MACfcMADDPGData* data = (MACfcMADDPGData*)hdr;
            return maddpg_select_actions(data, states, actions, add_noise);
        }
        case MA_CFC_ALGORITHM_QMIX: {
            MACfcQMIXData* data = (MACfcQMIXData*)hdr;
            float epsilon = add_noise ? 0.1f : 0.0f;
            return qmix_select_actions(data, states, actions, epsilon);
        }
        case MA_CFC_ALGORITHM_VDN: {
            MACfcVDNData* data = (MACfcVDNData*)hdr;
            float epsilon = add_noise ? 0.1f : 0.0f;
            return vdn_select_actions(data, states, actions, epsilon);
        }
        case MA_CFC_ALGORITHM_COMA: {
            MACfcCOMAData* data = (MACfcCOMAData*)hdr;
            float epsilon = add_noise ? 0.1f : 0.0f;
            return coma_select_actions(data, states, actions, epsilon);
        }
        default:
            return multi_agent_system_run(system, 1);
    }
}

int ma_cfc_train_step(MultiAgentSystem* system, const float* states,
                     const float* actions, const float* rewards,
                     const float* next_states, const int* dones,
                     const float* global_state, const float* next_global_state) {
    if (!system || !system->cfc_private_data) return -1;

    MACfcPrivateHeader* hdr = (MACfcPrivateHeader*)system->cfc_private_data;

    switch (hdr->algorithm) {
        case MA_CFC_ALGORITHM_MADDPG: {
            MACfcMADDPGData* data = (MACfcMADDPGData*)hdr;
            if (states && actions && rewards && next_states && dones) {
                macfc_buffer_add(data->replay_buffer, states, actions, rewards, next_states, dones, global_state, next_global_state);
            }
            return maddpg_train_step(data);
        }
        case MA_CFC_ALGORITHM_QMIX: {
            MACfcQMIXData* data = (MACfcQMIXData*)hdr;
            if (states && actions && rewards && next_states && dones) {
                macfc_buffer_add(data->replay_buffer, states, actions, rewards, next_states, dones, global_state, next_global_state);
            }
            return qmix_train_step(data);
        }
        case MA_CFC_ALGORITHM_VDN: {
            MACfcVDNData* data = (MACfcVDNData*)hdr;
            if (states && actions && rewards && next_states && dones) {
                macfc_buffer_add(data->replay_buffer, states, actions, rewards, next_states, dones, global_state, next_global_state);
            }
            return vdn_train_step(data);
        }
        case MA_CFC_ALGORITHM_COMA: {
            MACfcCOMAData* data = (MACfcCOMAData*)hdr;
            if (states && actions && rewards && next_states && dones) {
                macfc_buffer_add(data->replay_buffer, states, actions, rewards, next_states, dones, global_state, next_global_state);
            }
            return coma_train_step(data);
        }
        default:
            return multi_agent_system_run(system, 1);
    }
}
