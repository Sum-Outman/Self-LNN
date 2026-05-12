#ifndef SELFLNN_ROBOT_AGENT_H
#define SELFLNN_ROBOT_AGENT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGENT_NAME_MAX 64
#define AGENT_STATE_DIM 32
#define AGENT_ACTION_DIM 16
#define AGENT_MEMORY_SIZE 10000
#define AGENT_SKILL_MAX 50
#define AGENT_PLAN_MAX_STEPS 100
#define AGENT_KNOWLEDGE_MAX 500

typedef enum {
    AGENT_STATE_IDLE = 0,
    AGENT_STATE_LEARNING = 1,
    AGENT_STATE_EXECUTING = 2,
    AGENT_STATE_SELF_CORRECTING = 3,
    AGENT_STATE_PLANNING = 4,
    AGENT_STATE_REFLECTING = 5,
    AGENT_STATE_ERROR = -1
} AgentState;

typedef enum {
    LEARNING_MODE_NONE = 0,
    LEARNING_MODE_IMITATION = 1,
    LEARNING_MODE_REINFORCEMENT = 2,
    LEARNING_MODE_SELF_SUPERVISED = 3,
    LEARNING_MODE_EVOLUTIONARY = 4,
    LEARNING_MODE_CONTINUOUS = 5
} LearningMode;

typedef struct {
    float state[AGENT_STATE_DIM];
    float action[AGENT_ACTION_DIM];
    float reward;
    float next_state[AGENT_STATE_DIM];
    int done;
    float priority;
    float timestamp;
} AgentExperience;

typedef struct {
    AgentExperience buffer[AGENT_MEMORY_SIZE];
    int head;
    int count;
    int capacity;
} ReplayBuffer;

typedef struct {
    float* weights_ih;         /* 输入→隐藏层0 权重 [input_dim * hidden_dim] */
    float* weights_hh;         /* 隐藏层0→隐藏层1 [hidden_dim * hidden_dim] */
    float* weights_hh2;        /* 隐藏层1→隐藏层2 [hidden_dim * hidden_dim] */
    float* weights_ho;         /* 隐藏层2→输出 [hidden_dim * action_dim] */
    float* bias_h1;            /* 隐藏层0偏置 [hidden_dim] */
    float* bias_h2;            /* 隐藏层1偏置 [hidden_dim] */
    float* bias_h3;            /* 隐藏层2偏置 [hidden_dim] */
    float* bias_o;             /* 输出层偏置 [action_dim] */
    float learning_rate;
    float exploration_rate;
    float discount_factor;
    float lambda_trace;
    int input_dim;
    int output_dim;
    int hidden_dim;
} PolicyNetwork;

typedef struct {
    float state[AGENT_STATE_DIM];
    float target_state[AGENT_STATE_DIM];
    char description[256];
    int steps;
    int max_steps;
    float reward_threshold;
} AgentGoal;

typedef struct {
    float state_embedding[AGENT_STATE_DIM];
    char skill_name[64];
    float skill_parameters[AGENT_ACTION_DIM];
    int execution_count;
    float success_rate;
    float average_reward;
} AgentSkill;

typedef struct {
    char knowledge_key[128];
    char knowledge_value[512];
    float confidence;
    int access_count;
} KnowledgeEntry;

typedef struct {
    char name[AGENT_NAME_MAX];
    AgentState state;
    float state_vec[AGENT_STATE_DIM];
    LearningMode learning_mode;

    ReplayBuffer replay_buffer;
    PolicyNetwork policy;
    PolicyNetwork target_policy;

    AgentGoal current_goal;
    AgentSkill skills[AGENT_SKILL_MAX];
    int skill_count;

    KnowledgeEntry knowledge_base[AGENT_KNOWLEDGE_MAX];
    int knowledge_count;

    int enable_self_learning;
    int enable_self_evolution;
    int enable_self_correction;
    int enable_imitation_learning;
    int enable_autonomous_execution;

    float cumulative_reward;
    float episode_reward;
    int episode_count;
    int total_steps;

    float self_correction_threshold;
    float learning_progress;
    float evolution_rate;
    int plan_horizon;
    float plan_steps[AGENT_PLAN_MAX_STEPS][AGENT_ACTION_DIM];
    int plan_count;

    float state_history[100][AGENT_STATE_DIM];
    int history_count;
    float state_mean[AGENT_STATE_DIM];
    float state_std[AGENT_STATE_DIM];

    int self_awareness_level;
    float confidence_score;
    float curiosity_factor;
} RobotAgent;

typedef struct {
    LearningMode learning_mode;
    int enable_self_learning;
    int enable_self_evolution;
    int enable_self_correction;
    int enable_imitation_learning;
    int enable_autonomous_execution;
    float learning_rate;
    float exploration_rate;
    float discount_factor;
    float self_correction_threshold;
    int replay_capacity;
    int state_dim;
    int action_dim;
    int hidden_dim;
    int plan_horizon;
    float curiosity_factor;
} AgentConfig;

extern const AgentConfig AGENT_CONFIG_DEFAULT;

RobotAgent* robot_agent_create(const AgentConfig* config);
void robot_agent_free(RobotAgent* agent);
int robot_agent_init(RobotAgent* agent, const AgentConfig* config);

int robot_agent_set_goal(RobotAgent* agent, const float* target_state,
                          const char* description, float reward_threshold);
int robot_agent_observe(RobotAgent* agent, const float* state);
int robot_agent_act(RobotAgent* agent, float* action);
int robot_agent_learn(RobotAgent* agent, const float* state,
                       const float* action, float reward,
                       const float* next_state, int done);

int robot_agent_imitate(RobotAgent* agent,
                         const float* expert_states,
                         const float* expert_actions,
                         int num_demonstrations);
int robot_agent_self_correct(RobotAgent* agent);
int robot_agent_evolve(RobotAgent* agent);

int robot_agent_plan(RobotAgent* agent, const float* target_state,
                      int horizon);
int robot_agent_execute_plan(RobotAgent* agent, float* next_action);

int robot_agent_add_skill(RobotAgent* agent, const char* name,
                           const float* parameters);
int robot_agent_add_knowledge(RobotAgent* agent, const char* key,
                               const char* value, float confidence);
const char* robot_agent_query_knowledge(RobotAgent* agent, const char* key);

int robot_agent_reflect(RobotAgent* agent);
int robot_agent_update_self_awareness(RobotAgent* agent);
int robot_agent_get_status(const RobotAgent* agent, char* buffer, size_t size);
int robot_agent_save(const RobotAgent* agent, const char* filepath);
int robot_agent_load(RobotAgent* agent, const char* filepath);
int robot_agent_reset(RobotAgent* agent);

#ifdef __cplusplus
}
#endif

#endif
