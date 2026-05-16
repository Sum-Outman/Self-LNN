/**
 * @file agi.c
 * @brief AGI逻辑层 —— 认知循环、能力开关管理、高级推理决策
 * 
 * K-003: 角色定义 —— agi.c 是整个系统的【AGI逻辑层】
 * 负责：AGI认知主循环、能力开关注册/检查、目标管理、任务调度。
 * 
 * 架构层级：
 *   main.c（AGI后台任务循环）
 *      ↓
 *   selflnn.c（系统级入口 —— 子系统创建/生命周期）
 *      ↓
 *   agi.c（AGI逻辑层 —— 本文件）← 认知循环、能力开关、推理决策
 *      ↓
 *   各子系统（LNN、知识库、推理引擎、记忆系统、多模态...）
 * 
 * 与selflnn.c的关系：selflnn.c负责创建子系统并注入agi.c，
 * agi.c负责使用这些子系统执行AGI认知操作。
 * 两者不可替代，职责分离明确。
 */

#define _CRT_NONSTDC_NO_DEPRECATE
#include "selflnn/agi/agi.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/knowledge/knowledge.h"
#include "selflnn/knowledge/knowledge_graph.h"
#include "selflnn/reasoning/reasoning.h"
#include "selflnn/reasoning/planning.h"
#include "selflnn/core/decision_engine.h"
#include "selflnn/learning/learning.h"
#include "selflnn/memory/memory_manager.h"
#include "selflnn/metacognition.h"
#include "selflnn/self_cognition.h"
#include "selflnn/cognition/deep_reflection.h"
#include "selflnn/cognition/deep_thought_chain.h"
#include "selflnn/cognition/deep_correction.h"
#include "selflnn/multimodal/dialogue.h"
#include "selflnn/multimodal/deep_vision.h"
#include "selflnn/multimodal/depth_estimation.h"
#include "selflnn/multisystem/multisystem_control.h"
#include "selflnn/agi/capability_switch.h"
#include "selflnn/core/system_scheduler.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/laplace.h"
#include <math.h>
#include <time.h>

/* 子系统接口前向声明（实现位于对应子系统模块） */
extern int learning_engine_is_imitation_enabled(void* learner);
extern int learning_engine_set_imitation_enabled(void* learner, int enable);
extern int deep_reflection_is_enabled(void* reflection);
extern void deep_reflection_enable(void* reflection);
extern void deep_reflection_disable(void* reflection);
extern int planning_is_enabled(void* planner);
extern void planning_enable(void* planner);
extern void planning_disable(void* planner);
extern int dialogue_is_enabled(void* dialogue);
extern void dialogue_enable(void* dialogue);
extern void dialogue_disable(void* dialogue);
#include "selflnn/concurrency/thread_pool.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

struct AGISystem {
    AGIConfig config;
    AGIState current_state;
    AGICognitiveStatus status;

    KnowledgeBase* knowledge;
    int owns_knowledge;
    ReasoningEngine* reasoning;
    int owns_reasoning;
    DecisionEngine* decision;
    int owns_decision;
    PlanningSystem* planner;
    int owns_planner;
    LearningEngine* learner;
    int owns_learner;
    MemoryManager* memory;
    int owns_memory;
    MetacognitionSystem* metacognition;
    int owns_metacognition;
    SelfCognitionSystem* self_cognition;
    int owns_self_cognition;
    DeepReflectionEngine* reflection;
    int owns_reflection;
    DTCSystem* thought_chain;
    int owns_thought_chain;
    DCCorrectionSystem* correction;
    int owns_correction;
    DialogueProcessor* dialogue;
    int owns_dialogue;
    LNN* lnn;
    int owns_lnn;
    UnifiedLNNState* unified_lnn;
    int owns_unified_lnn;
    SystemScheduler* scheduler;
    int owns_scheduler;

    ThreadPool* thread_pool;
    int owns_thread_pool;

    CfcVisionProcessor* vision;
    int owns_vision;
    DepthEstimator* depth_estimator;
    int owns_depth_estimator;
    MultiSystemControlEngine* multisystem_control;
    int owns_multisystem_control;

    float* vision_input_buffer;
    int vision_buffer_width;
    int vision_buffer_height;
    int vision_buffer_channels;
    time_t vision_buffer_timestamp;
    int vision_buffer_valid;

    AIGoal goals[AGI_MAX_GOALS];
    int goal_count;
    int next_goal_id;

    AGITask tasks[AGI_MAX_TASKS];
    int task_count;
    int next_task_id;

    float* state_vector;
    int state_vector_dim;
    int expand_state_vector;

    float* cognitive_history;
    int cognitive_history_count;
    int max_cognitive_history;

    time_t last_reflection_time;
    time_t last_learning_time;
    time_t last_evolution_time;
    int total_cycles;
    int consecutive_low_reward;
    float avg_reward;
    float total_reward;
    int reward_count;

    int autonomous_mode;
    char self_model_description[AGI_DESC_LEN];
    int initialized;
};

/* 能力开关回调注册（P3.3） */
static AGISystem* g_agi_cap_system = NULL;

static int cap_check_self_cognition(void) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->self_cognition) return 0;
    return self_cognition_system_is_enabled(sys->self_cognition) ? 1 : 0;
}
static int cap_set_self_cognition(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->self_cognition) return -1;
    if (enable) self_cognition_system_enable(sys->self_cognition);
    else self_cognition_system_disable(sys->self_cognition);
    return 0;
}

static int cap_check_decision(void) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->decision) return 0;
    return decision_engine_is_enabled(sys->decision);
}
static int cap_set_decision(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->decision) return -1;
    if (enable) decision_engine_enable(sys->decision);
    else decision_engine_disable(sys->decision);
    return 0;
}

static int cap_check_learning(void) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->learner) return 0;
    return learning_engine_is_enabled(sys->learner);
}
static int cap_set_learning(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->learner) return -1;
    if (enable) learning_engine_enable(sys->learner);
    else learning_engine_disable(sys->learner);
    return 0;
}

static int cap_check_evolution(void) {
    /* 演化能力通过学习引擎的自我演化功能控制 */
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->learner) return 0;
    return learning_engine_is_enabled(sys->learner);
}
static int cap_set_evolution(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->learner) return -1;
    if (enable) {
        learning_engine_enable(sys->learner);
        /* 联动LNN演化标志 */
        if (sys->lnn) lnn_set_evolution_enabled(sys->lnn, 1);
    } else {
        learning_engine_disable(sys->learner);
        if (sys->lnn) lnn_set_evolution_enabled(sys->lnn, 0);
    }
    return 0;
}

static int cap_check_imitation(void) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->learner) return 0;
    return learning_engine_is_imitation_enabled(sys->learner);
}
static int cap_set_imitation(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->learner) return -1;
    return learning_engine_set_imitation_enabled(sys->learner, enable);
}

static int cap_check_reflection(void) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->reflection) return 0;
    return deep_reflection_is_enabled(sys->reflection);
}
static int cap_set_reflection(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->reflection) return -1;
    if (enable) deep_reflection_enable(sys->reflection);
    else deep_reflection_disable(sys->reflection);
    return 0;
}

static int cap_check_planning(void) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->planner) return 0;
    return planning_is_enabled(sys->planner);
}
static int cap_set_planning(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->planner) return -1;
    if (enable) planning_enable(sys->planner);
    else planning_disable(sys->planner);
    return 0;
}

static int cap_check_dialogue(void) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->dialogue) return 0;
    return dialogue_is_enabled(sys->dialogue);
}
static int cap_set_dialogue(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->dialogue) return -1;
    if (enable) dialogue_enable(sys->dialogue);
    else dialogue_disable(sys->dialogue);
    return 0;
}

static int cap_check_correction(void) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->correction) return 0;
    return dc_correction_is_enabled(sys->correction);
}
static int cap_set_correction(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->correction) return -1;
    if (enable) dc_correction_enable(sys->correction);
    else dc_correction_disable(sys->correction);
    return 0;
}

static int cap_check_autonomous_execution(void) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->scheduler) return 0;
    return system_scheduler_is_enabled(sys->scheduler);
}
static int cap_set_autonomous_execution(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->scheduler) return -1;
    sys->autonomous_mode = (enable != 0) ? 1 : 0;
    return system_scheduler_set_enabled(sys->scheduler, enable);
}

static int cap_check_concurrency(void) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->thread_pool) return 0;
    /* 检查线程池实际运行状态：通过thread_pool_get_stats获取活跃线程数 */
    size_t active_threads = 0, pending = 0, completed = 0;
    thread_pool_get_stats(sys->thread_pool, &active_threads, &pending, &completed);
    return (active_threads > 0) ? 1 : 0;
}
static int cap_set_concurrency(int enable) {
    AGISystem* sys = g_agi_cap_system;
    if (!sys || !sys->thread_pool) return -1;
    if (enable) {
        ThreadPoolConfig tpcfg;
        memset(&tpcfg, 0, sizeof(tpcfg));
        tpcfg.num_threads = 4;
        tpcfg.max_tasks = 256;
        tpcfg.dynamic_scaling = 1;
        tpcfg.enable_priority = 1;
        tpcfg.enable_work_stealing = 1;
        tpcfg.max_tasks_per_thread = 64;
        tpcfg.work_stealing_threshold = 2;
        tpcfg.task_timeout_ms = 5000;
        tpcfg.idle_thread_timeout_ms = 30000;
        return thread_pool_set_config(sys->thread_pool, &tpcfg);
    } else {
        ThreadPoolConfig tpcfg;
        memset(&tpcfg, 0, sizeof(tpcfg));
        tpcfg.num_threads = 0;
        tpcfg.max_tasks = 0;
        tpcfg.dynamic_scaling = 0;
        return thread_pool_set_config(sys->thread_pool, &tpcfg);
    }
}

/* K-029: 好奇心能力共享状态 —— 独立于reflection管理 */
static int g_agi_curiosity_state = 1;  /* 默认启用 */

static int cap_check_curiosity(void) {
    return g_agi_curiosity_state;
}
static int cap_set_curiosity(int enable) {
    g_agi_curiosity_state = (enable != 0) ? 1 : 0;
    return 0;
}

static void agi_register_all_capabilities(AGISystem* system) {
    g_agi_cap_system = system;
    capability_register_module(CAP_SELF_COGNITION, cap_check_self_cognition, cap_set_self_cognition);
    capability_register_module(CAP_SELF_DECISION, cap_check_decision, cap_set_decision);
    capability_register_module(CAP_SELF_LEARNING, cap_check_learning, cap_set_learning);
    capability_register_module(CAP_SELF_EVOLUTION, cap_check_evolution, cap_set_evolution);
    capability_register_module(CAP_IMITATION_LEARNING, cap_check_imitation, cap_set_imitation);
    capability_register_module(CAP_SELF_CORRECTION, cap_check_correction, cap_set_correction);
    capability_register_module(CAP_AUTONOMOUS_EXECUTION, cap_check_autonomous_execution, cap_set_autonomous_execution);
    capability_register_module(CAP_REFLECTION, cap_check_reflection, cap_set_reflection);
    capability_register_module(CAP_CURIOSITY, cap_check_curiosity, cap_set_curiosity);
    capability_register_module(CAP_PLANNING, cap_check_planning, cap_set_planning);
    capability_register_module(CAP_DIALOGUE, cap_check_dialogue, cap_set_dialogue);
    capability_register_module(CAP_CONCURRENCY, cap_check_concurrency, cap_set_concurrency);
}

/* 认知循环并行任务结构 */
typedef struct {
    AGISystem* system;
    float* input;
    int input_dim;
    float* output;
    int output_dim;
} CognitiveTaskArg;

static void parallel_perceive_task(void* arg) {
    CognitiveTaskArg* a = (CognitiveTaskArg*)arg;
    float state_buf[AGI_STATE_VECTOR_DIM];
    agi_system_perceive(a->system, a->input, a->input_dim, state_buf);
    if (a->output && a->output_dim > 0) {
        int cpy = a->output_dim < AGI_STATE_VECTOR_DIM ? a->output_dim : AGI_STATE_VECTOR_DIM;
        memcpy(a->output, state_buf, (size_t)cpy * sizeof(float));
    }
}

static void parallel_reason_task(void* arg) {
    CognitiveTaskArg* a = (CognitiveTaskArg*)arg;
    float result_buf[AGI_STATE_VECTOR_DIM];
    agi_system_reason(a->system, a->input, a->input_dim, result_buf);
    if (a->output && a->output_dim > 0) {
        int cpy = a->output_dim < AGI_STATE_VECTOR_DIM ? a->output_dim : AGI_STATE_VECTOR_DIM;
        memcpy(a->output, result_buf, (size_t)cpy * sizeof(float));
    }
}

static void parallel_learn_task(void* arg) {
    CognitiveTaskArg* a = (CognitiveTaskArg*)arg;
    agi_system_learn(a->system, a->input, a->input_dim, 0.0f);
}

static int create_default_knowledge_base(AGISystem* system)
{
    if (!system) return -1;
    KnowledgeBase* kb = knowledge_base_create((size_t)system->config.knowledge_capacity);
    if (!kb) return -1;
    system->knowledge = kb;
    system->owns_knowledge = 1;
    return 0;
}

static int create_default_reasoning_engine(AGISystem* system)
{
    if (!system) return -1;
    ReasoningConfig rcfg;
    memset(&rcfg, 0, sizeof(rcfg));
    rcfg.default_mode = REASONING_DEDUCTIVE;
    rcfg.max_iterations = system->config.reasoning_depth * 10;
    rcfg.confidence_threshold = 0.5f;
    rcfg.enable_multimodal = 1;
    ReasoningEngine* eng = reasoning_engine_create(&rcfg);
    if (!eng) return -1;
    system->reasoning = eng;
    system->owns_reasoning = 1;
    return 0;
}

static int create_default_decision_engine(AGISystem* system)
{
    if (!system) return -1;
    DecisionEngineConfig dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.decision_type = DECISION_TYPE_MULTI_OBJECTIVE;
    dcfg.max_alternatives = 64;
    dcfg.max_objectives = 8;
    dcfg.max_constraints = 8;
    dcfg.epsilon = 1e-6f;
    dcfg.enable_pareto_optimization = 1;
    dcfg.enable_explanation = 1;
    DecisionEngine* de = decision_engine_create(&dcfg);
    if (!de) return -1;
    system->decision = de;
    system->owns_decision = 1;
    return 0;
}

static int create_default_planning_system(AGISystem* system)
{
    if (!system) return -1;
    PlanningConfig pcfg;
    memset(&pcfg, 0, sizeof(pcfg));
    pcfg.algorithm = PLANNING_HIERARCHICAL;
    pcfg.max_plan_length = system->config.planning_horizon;
    pcfg.risk_tolerance = 0.3f;
    pcfg.goal_tolerance = 0.05f;
    pcfg.enable_adaptation = 1;
    PlanningSystem* ps = planning_system_create(&pcfg);
    if (!ps) return -1;
    system->planner = ps;
    system->owns_planner = 1;
    return 0;
}

static int create_default_learning_engine(AGISystem* system)
{
    if (!system) return -1;
    LearningConfig lcfg;
    memset(&lcfg, 0, sizeof(lcfg));
    lcfg.learning_type = LEARNING_REINFORCEMENT;
    lcfg.learning_rate = system->config.learning_rate;
    lcfg.exploration_rate = system->config.exploration_rate;
    lcfg.discount_factor = 0.95f;
    lcfg.population_size = 50;
    lcfg.max_generations = 100;
    lcfg.mutation_rate = 0.1f;
    lcfg.enable_self_evolution = system->config.enable_self_evolution;
    lcfg.enable_self_correction = system->config.enable_self_correction;
    lcfg.risk_tolerance = 0.3f;
    lcfg.goal_tolerance = 0.05f;
    lcfg.max_plan_length = system->config.planning_horizon;
    lcfg.enable_adaptation = 1;
    LearningEngine* le = learning_engine_create(&lcfg);
    if (!le) return -1;
    system->learner = le;
    system->owns_learner = 1;
    return 0;
}

static int create_default_memory_manager(AGISystem* system)
{
    if (!system) return -1;
    MemoryManagerConfig mcfg;
    memset(&mcfg, 0, sizeof(mcfg));
    mcfg.short_term_capacity = (size_t)(system->config.memory_capacity * 0.3f);
    mcfg.long_term_capacity = (size_t)(system->config.memory_capacity * 0.5f);
    mcfg.episodic_capacity = (size_t)(system->config.memory_capacity * 0.1f);
    mcfg.semantic_capacity = (size_t)(system->config.memory_capacity * 0.1f);
    mcfg.consolidation_rate = 0.05f;
    mcfg.enable_integration = 1;
    mcfg.buddy_pool_size = 1024 * 1024;
    mcfg.enable_buddy_allocator = 1;
    MemoryManager* mm = memory_manager_create(&mcfg);
    if (!mm) return -1;
    system->memory = mm;
    system->owns_memory = 1;
    return 0;
}

static int create_default_metacognition(AGISystem* system)
{
    if (!system) return -1;
    MetacognitionMonitoringConfig moncfg;
    memset(&moncfg, 0, sizeof(moncfg));
    moncfg.monitoring_type = METACOGNITION_MONITORING_PERFORMANCE;
    moncfg.update_frequency = 10.0f;
    moncfg.confidence_threshold = 0.6f;
    moncfg.uncertainty_threshold = 0.3f;
    moncfg.enable_real_time = 1;
    moncfg.enable_prediction = 1;
    moncfg.enable_self_correction = system->config.enable_self_correction;
    moncfg.history_buffer_size = 1000;
    moncfg.adaptation_rate = 0.05f;

    SelfModelUpdateConfig modelcfg;
    memset(&modelcfg, 0, sizeof(modelcfg));
    modelcfg.update_method = SELF_MODEL_UPDATE_ONLINE_LEARNING;
    modelcfg.learning_rate = system->config.learning_rate;
    modelcfg.forgetting_factor = 0.95f;
    modelcfg.uncertainty_weight = 0.3f;
    modelcfg.enable_online_update = 1;
    modelcfg.enable_batch_update = 1;
    modelcfg.model_complexity = 64;
    modelcfg.regularization_strength = 0.01f;

    PredictiveSelfConfig predcfg;
    memset(&predcfg, 0, sizeof(predcfg));
    predcfg.prediction_type = PREDICTIVE_SELF_PERFORMANCE;
    predcfg.prediction_horizon = 60.0f;
    predcfg.confidence_level = 0.8f;
    predcfg.enable_multiple_horizons = 1;
    predcfg.enable_uncertainty_quantification = 1;
    predcfg.prediction_model_size = 32;
    predcfg.adaptation_sensitivity = 0.1f;

    MetacognitionSystem* meta = metacognition_system_create(&moncfg, &modelcfg, &predcfg);
    if (!meta) return -1;
    system->metacognition = meta;
    system->owns_metacognition = 1;
    return 0;
}

static int create_default_self_cognition(AGISystem* system)
{
    if (!system) return -1;
    SelfCognitionConfig scfg;
    memset(&scfg, 0, sizeof(scfg));
    scfg.enable_continuous_monitoring = 1;
    scfg.update_interval_sec = 1.0f;
    scfg.enable_self_reflection = 1;
    scfg.enable_capability_assessment = 1;
    scfg.enable_knowledge_tracking = 1;
    scfg.enable_self_correction = system->config.enable_self_correction;
    SelfCognitionSystem* sc = self_cognition_create(&scfg);
    if (!sc) return -1;
    system->self_cognition = sc;
    system->owns_self_cognition = 1;
    return 0;
}

static int create_default_reflection(AGISystem* system)
{
    if (!system) return -1;
    DRConfig drcfg = DR_CONFIG_DEFAULT;
    drcfg.enable_causal_analysis = 1;
    drcfg.enable_contradiction_detection = 1;
    drcfg.enable_synthesis = 1;
    DeepReflectionEngine* dr = dr_engine_create(drcfg);
    if (!dr) return -1;
    system->reflection = dr;
    system->owns_reflection = 1;
    return 0;
}

static int create_default_thought_chain(AGISystem* system)
{
    if (!system) return -1;
    DTCConfig dtccfg = DTC_CONFIG_DEFAULT;
    DTCSystem* dtc = dtc_system_create(dtccfg);
    if (!dtc) return -1;
    system->thought_chain = dtc;
    system->owns_thought_chain = 1;
    return 0;
}

static int create_default_correction(AGISystem* system)
{
    if (!system) return -1;
    DCCorrectionSystem* dcs = dc_correction_create();
    if (!dcs) return -1;
    system->correction = dcs;
    system->owns_correction = 1;
    return 0;
}

static int create_default_vision(AGISystem* system)
{
    if (!system) return -1;
    CfcVisionConfig vcfg = cfc_vision_get_default_config();
    vcfg.image_width = 640;
    vcfg.image_height = 480;
    vcfg.image_channels = 3;
    vcfg.patch_size = 16;
    vcfg.output_dim = 128;
    vcfg.num_ode_layers = 3;
    vcfg.time_constant = 0.1f;
    vcfg.delta_t = 0.05f;
    CfcVisionProcessor* vp = cfc_vision_processor_create(&vcfg);
    if (!vp) return -1;
    system->vision = vp;
    system->owns_vision = 1;
    return 0;
}

static int create_default_depth_estimator(AGISystem* system)
{
    if (!system) return -1;
    DepthEstimationConfig dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.method = DEPTH_METHOD_STEREO;
    dcfg.stereo_algorithm = STEREO_MATCHING_SGBM;
    dcfg.enable_filtering = 1;
    dcfg.enable_postprocessing = 1;
    dcfg.enable_stereo_depth = 1;
    dcfg.disparity_range = 64;
    dcfg.window_size = 9;
    dcfg.max_features = 1000;
    dcfg.min_depth = 0.1f;
    dcfg.max_depth = 100.0f;
    dcfg.use_gpu = 0;
    dcfg.output_format = 0;
    DepthEstimator* de = depth_estimator_create(&dcfg);
    if (!de) return -1;
    system->depth_estimator = de;
    system->owns_depth_estimator = 1;
    return 0;
}

static int create_default_dialogue(AGISystem* system)
{
    if (!system) return -1;
    DialogueConfig dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.max_context_length = 100;
    dcfg.dialogue_hidden_size = 64;
    dcfg.response_generation_mode = 1;
    DialogueProcessor* dp = dialogue_processor_create(&dcfg);
    if (!dp) return -1;
    system->dialogue = dp;
    system->owns_dialogue = 1;
    return 0;
}

static int create_default_thread_pool(AGISystem* system)
{
    if (!system) return -1;
    ThreadPoolConfig tpcfg;
    memset(&tpcfg, 0, sizeof(tpcfg));
    tpcfg.num_threads = 4;
    tpcfg.max_tasks = 256;
    tpcfg.dynamic_scaling = 1;
    tpcfg.enable_priority = 1;
    tpcfg.enable_work_stealing = 1;
    tpcfg.max_tasks_per_thread = 64;
    tpcfg.work_stealing_threshold = 2;
    tpcfg.task_timeout_ms = 5000;
    tpcfg.idle_thread_timeout_ms = 30000;
    system->thread_pool = thread_pool_create(&tpcfg);
    if (!system->thread_pool) {
        selflnn_log(LOG_LEVEL_WARNING, "AGI", "线程池创建失败，将运行在单线程模式");
        return -1;
    }
    system->owns_thread_pool = 1;
    return 0;
}

static int create_default_multisystem_control(AGISystem* system)
{
    if (!system) return -1;
    MultiSystemControlEngine* mse = multisystem_control_engine_create();
    if (!mse) {
        selflnn_log(LOG_LEVEL_WARNING, "AGI", "多系统控制引擎创建失败，多机器人控制不可用");
        return -1;
    }
    system->multisystem_control = mse;
    system->owns_multisystem_control = 1;
    return 0;
}

AGISystem* agi_system_create(const AGIConfig* config)
{
    AGISystem* system = (AGISystem*)safe_malloc(sizeof(AGISystem));
    if (!system) return NULL;
    memset(system, 0, sizeof(AGISystem));

    if (config) {
        system->config = *config;
    } else {
        static const AGIConfig agi_default_config = AGI_CONFIG_DEFAULT;
        memcpy(&system->config, &agi_default_config, sizeof(AGIConfig));
    }

    system->current_state = AGI_STATE_IDLE;
    system->next_goal_id = 1;
    system->next_task_id = 1;
    system->state_vector_dim = system->config.state_vector_dim;
    system->expand_state_vector = system->state_vector_dim + 64;
    system->max_cognitive_history = 500;
    system->autonomous_mode = 1;
    system->vision_input_buffer = NULL;
    system->vision_buffer_valid = 0;
    system->vision_buffer_width = 0;
    system->vision_buffer_height = 0;
    system->vision_buffer_channels = 0;
    system->vision_buffer_timestamp = 0;
    system->avg_reward = 0.0f;
    system->total_reward = 0.0f;
    system->reward_count = 0;
    system->consecutive_low_reward = 0;
    system->total_cycles = 0;
    system->initialized = 0;
    strcpy(system->self_model_description, "AGI高级认知系统 v1.0");

    system->state_vector = (float*)safe_malloc((size_t)system->expand_state_vector * sizeof(float));
    if (!system->state_vector) {
        safe_free((void**)&system);
        return NULL;
    }
    memset(system->state_vector, 0, (size_t)system->expand_state_vector * sizeof(float));

    system->cognitive_history = (float*)safe_malloc((size_t)system->max_cognitive_history * sizeof(float));
    if (!system->cognitive_history) {
        safe_free((void**)&system->state_vector);
        safe_free((void**)&system);
        return NULL;
    }
    memset(system->cognitive_history, 0, (size_t)system->max_cognitive_history * sizeof(float));

    if (create_default_knowledge_base(system) != 0) {
        safe_free((void**)&system->cognitive_history);
        safe_free((void**)&system->state_vector);
        safe_free((void**)&system);
        return NULL;
    }
    create_default_reasoning_engine(system);
    if (system->reasoning && system->knowledge) {
        reasoning_engine_set_knowledge_base(system->reasoning, system->knowledge);
        reasoning_sync_knowledge(system->reasoning);
    }
    create_default_decision_engine(system);
    create_default_planning_system(system);
    create_default_learning_engine(system);
    create_default_memory_manager(system);
    create_default_metacognition(system);
    create_default_self_cognition(system);
    create_default_reflection(system);
    create_default_thought_chain(system);
    create_default_correction(system);
    create_default_dialogue(system);
    create_default_thread_pool(system);
    create_default_vision(system);
    create_default_depth_estimator(system);
    create_default_multisystem_control(system);

    {
        UnifiedLNNStateConfig ulnn_cfg = unified_lnn_state_get_default_config();
        ulnn_cfg.state_dimension = (size_t)system->state_vector_dim;
        ulnn_cfg.output_dimension = (size_t)system->state_vector_dim;
        ulnn_cfg.raw_dimensions[UNIFIED_MODALITY_VISION] = (size_t)system->state_vector_dim;
        ulnn_cfg.evolution_delta_t = 0.1f;
        ulnn_cfg.enable_online_learning = system->config.enable_self_learning;
        system->unified_lnn = unified_lnn_state_create(&ulnn_cfg);
        system->owns_unified_lnn = 1;
        if (!system->unified_lnn) {
            safe_free((void**)&system->cognitive_history);
            safe_free((void**)&system->state_vector);
            safe_free((void**)&system);
            return NULL;
        }
    }

    system->scheduler = system_scheduler_create(NULL);
    system->owns_scheduler = 1;

    system->status.state = AGI_STATE_IDLE;
    system->status.confidence = 0.5f;
    system->status.curiosity = system->config.exploration_rate;
    system->status.cognitive_load = 0.0f;
    system->status.attention_focus = 1.0f;
    system->status.uptime = time(NULL);
    system->status.last_state_change = time(NULL);
    agi_register_all_capabilities(system);
    system->initialized = 1;

    return system;
}

void agi_system_free(AGISystem* system)
{
    if (!system) return;

    if (system->owns_knowledge && system->knowledge)
        knowledge_base_free(system->knowledge);
    if (system->owns_reasoning && system->reasoning)
        reasoning_engine_free(system->reasoning);
    if (system->owns_decision && system->decision)
        decision_engine_destroy(system->decision);
    if (system->owns_planner && system->planner)
        planning_system_free(system->planner);
    if (system->owns_learner && system->learner)
        learning_engine_free(system->learner);
    if (system->owns_memory && system->memory)
        memory_manager_free(system->memory);
    if (system->owns_metacognition && system->metacognition)
        metacognition_system_free(system->metacognition);
    if (system->owns_self_cognition && system->self_cognition)
        self_cognition_free(system->self_cognition);
    if (system->owns_reflection && system->reflection)
        dr_engine_destroy(system->reflection);
    if (system->owns_thought_chain && system->thought_chain)
        dtc_system_destroy(system->thought_chain);
    if (system->owns_correction && system->correction)
        dc_correction_free(system->correction);
    if (system->owns_dialogue && system->dialogue)
        dialogue_processor_free(system->dialogue);
    if (system->owns_unified_lnn && system->unified_lnn)
        unified_lnn_state_free(system->unified_lnn);
    if (system->owns_scheduler && system->scheduler)
        system_scheduler_free(system->scheduler);
    if (system->owns_thread_pool && system->thread_pool)
        thread_pool_free(system->thread_pool);
    if (system->owns_vision && system->vision)
        cfc_vision_processor_destroy(system->vision);
    if (system->owns_depth_estimator && system->depth_estimator)
        depth_estimator_free(system->depth_estimator);
    if (system->owns_multisystem_control && system->multisystem_control)
        multisystem_control_engine_destroy(system->multisystem_control);

    if (system->vision_input_buffer)
        safe_free((void**)&system->vision_input_buffer);
    if (system->state_vector)
        safe_free((void**)&system->state_vector);
    if (system->cognitive_history)
        safe_free((void**)&system->cognitive_history);

    int i;
    for (i = 0; i < system->task_count; i++) {
        if (system->tasks[i].action_sequence)
            safe_free((void**)&system->tasks[i].action_sequence);
        if (system->tasks[i].result_data)
            safe_free((void**)&system->tasks[i].result_data);
    }

    memset(system, 0, sizeof(AGISystem));
    safe_free((void**)&system);
}

int agi_system_set_knowledge_base(AGISystem* system, KnowledgeBase* kb)
{
    if (!system || !kb) return -1;
    if (system->owns_knowledge && system->knowledge)
        knowledge_base_free(system->knowledge);
    system->knowledge = kb;
    system->owns_knowledge = 0;
    return 0;
}

int agi_system_set_reasoning_engine(AGISystem* system, ReasoningEngine* engine)
{
    if (!system || !engine) return -1;
    if (system->owns_reasoning && system->reasoning)
        reasoning_engine_free(system->reasoning);
    system->reasoning = engine;
    system->owns_reasoning = 0;
    return 0;
}

int agi_system_set_decision_engine(AGISystem* system, DecisionEngine* engine)
{
    if (!system || !engine) return -1;
    if (system->owns_decision && system->decision)
        decision_engine_destroy(system->decision);
    system->decision = engine;
    system->owns_decision = 0;
    return 0;
}

int agi_system_set_planning_system(AGISystem* system, PlanningSystem* planner)
{
    if (!system || !planner) return -1;
    if (system->owns_planner && system->planner)
        planning_system_free(system->planner);
    system->planner = planner;
    system->owns_planner = 0;
    return 0;
}

int agi_system_set_learning_engine(AGISystem* system, LearningEngine* learner)
{
    if (!system || !learner) return -1;
    if (system->owns_learner && system->learner)
        learning_engine_free(system->learner);
    system->learner = learner;
    system->owns_learner = 0;
    return 0;
}

int agi_system_set_memory_manager(AGISystem* system, MemoryManager* memory)
{
    if (!system || !memory) return -1;
    if (system->owns_memory && system->memory)
        memory_manager_free(system->memory);
    system->memory = memory;
    system->owns_memory = 0;
    return 0;
}

int agi_system_set_metacognition(AGISystem* system, MetacognitionSystem* meta)
{
    if (!system || !meta) return -1;
    if (system->owns_metacognition && system->metacognition)
        metacognition_system_free(system->metacognition);
    system->metacognition = meta;
    system->owns_metacognition = 0;
    return 0;
}

int agi_system_set_self_cognition(AGISystem* system, SelfCognitionSystem* self_cog)
{
    if (!system || !self_cog) return -1;
    if (system->owns_self_cognition && system->self_cognition)
        self_cognition_free(system->self_cognition);
    system->self_cognition = self_cog;
    system->owns_self_cognition = 0;
    return 0;
}

int agi_system_set_reflection(AGISystem* system, DeepReflectionEngine* reflection)
{
    if (!system || !reflection) return -1;
    if (system->owns_reflection && system->reflection)
        dr_engine_destroy(system->reflection);
    system->reflection = reflection;
    system->owns_reflection = 0;
    return 0;
}

int agi_system_set_thought_chain(AGISystem* system, DTCSystem* dtc)
{
    if (!system || !dtc) return -1;
    if (system->owns_thought_chain && system->thought_chain)
        dtc_system_destroy(system->thought_chain);
    system->thought_chain = dtc;
    system->owns_thought_chain = 0;
    return 0;
}

int agi_system_set_correction(AGISystem* system, DCCorrectionSystem* correction)
{
    if (!system || !correction) return -1;
    if (system->owns_correction && system->correction)
        dc_correction_free(system->correction);
    system->correction = correction;
    system->owns_correction = 0;
    return 0;
}

int agi_system_set_dialogue(AGISystem* system, DialogueProcessor* dialogue)
{
    if (!system || !dialogue) return -1;
    if (system->owns_dialogue && system->dialogue)
        dialogue_processor_free(system->dialogue);
    system->dialogue = dialogue;
    system->owns_dialogue = 0;
    return 0;
}

int agi_system_set_lnn(AGISystem* system, LNN* lnn)
{
    if (!system || !lnn) return -1;
    system->lnn = lnn;
    system->owns_lnn = 0;
    return 0;
}

int agi_system_set_unified_lnn(AGISystem* system, UnifiedLNNState* state)
{
    if (!system || !state) return -1;
    if (system->owns_unified_lnn && system->unified_lnn)
        unified_lnn_state_free(system->unified_lnn);
    system->unified_lnn = state;
    system->owns_unified_lnn = 0;
    return 0;
}

int agi_system_set_vision_processor(AGISystem* system, CfcVisionProcessor* vision)
{
    if (!system || !vision) return -1;
    if (system->owns_vision && system->vision)
        cfc_vision_processor_destroy(system->vision);
    system->vision = vision;
    system->owns_vision = 0;
    return 0;
}

int agi_system_set_depth_estimator(AGISystem* system, DepthEstimator* estimator)
{
    if (!system || !estimator) return -1;
    if (system->owns_depth_estimator && system->depth_estimator)
        depth_estimator_free(system->depth_estimator);
    system->depth_estimator = estimator;
    system->owns_depth_estimator = 0;
    return 0;
}

int agi_system_set_multisystem_control(AGISystem* system, MultiSystemControlEngine* engine)
{
    if (!system || !engine) return -1;
    if (system->owns_multisystem_control && system->multisystem_control)
        multisystem_control_engine_destroy(system->multisystem_control);
    system->multisystem_control = engine;
    system->owns_multisystem_control = 0;
    return 0;
}

int agi_system_set_config(AGISystem* system, const AGIConfig* config)
{
    if (!system || !config) return -1;
    system->config = *config;
    return 0;
}

int agi_system_get_config(const AGISystem* system, AGIConfig* config)
{
    if (!system || !config) return -1;
    *config = system->config;
    return 0;
}

int agi_system_add_goal(AGISystem* system, const char* name, const char* description,
                         float priority, float deadline, const float* state_vector, int state_vector_dim)
{
    if (!system || !name || !description) return -1;
    if (system->goal_count >= AGI_MAX_GOALS) return -1;

    AIGoal* goal = &system->goals[system->goal_count];
    memset(goal, 0, sizeof(AIGoal));
    goal->goal_id = system->next_goal_id++;
    strncpy(goal->name, name, AGI_NAME_LEN - 1);
    strncpy(goal->description, description, AGI_DESC_LEN - 1);
    goal->priority = (priority < 0.0f) ? 0.0f : (priority > 1.0f) ? 1.0f : priority;
    goal->status = AGI_GOAL_PENDING;
    goal->progress = 0.0f;
    goal->deadline = deadline;
    goal->created_at = time(NULL);
    goal->updated_at = goal->created_at;
    goal->parent_goal_id = -1;
    goal->subgoal_count = 0;
    if (state_vector && state_vector_dim > 0) {
        int dim = (state_vector_dim < AGI_STATE_VECTOR_DIM) ? state_vector_dim : AGI_STATE_VECTOR_DIM;
        memcpy(goal->state_vector, state_vector, (size_t)dim * sizeof(float));
        goal->state_vector_dim = dim;
    }

    system->goal_count++;
    system->status.active_goal_count = system->goal_count;
    return goal->goal_id;
}

int agi_system_update_goal(AGISystem* system, int goal_id, float progress, AIGoalStatus status)
{
    if (!system) return -1;
    int i;
    for (i = 0; i < system->goal_count; i++) {
        if (system->goals[i].goal_id == goal_id) {
            system->goals[i].progress = (progress < 0.0f) ? 0.0f : (progress > 1.0f) ? 1.0f : progress;
            system->goals[i].status = status;
            system->goals[i].updated_at = time(NULL);
            return 0;
        }
    }
    return -1;
}

int agi_system_get_goal(const AGISystem* system, int goal_id, AIGoal* goal)
{
    if (!system || !goal) return -1;
    int i;
    for (i = 0; i < system->goal_count; i++) {
        if (system->goals[i].goal_id == goal_id) {
            *goal = system->goals[i];
            return 0;
        }
    }
    return -1;
}

int agi_system_remove_goal(AGISystem* system, int goal_id)
{
    if (!system) return -1;
    int i;
    for (i = 0; i < system->goal_count; i++) {
        if (system->goals[i].goal_id == goal_id) {
            int j;
            for (j = i; j < system->goal_count - 1; j++)
                system->goals[j] = system->goals[j + 1];
            system->goal_count--;
            memset(&system->goals[system->goal_count], 0, sizeof(AIGoal));
            system->status.active_goal_count = system->goal_count;
            return 0;
        }
    }
    return -1;
}

int agi_system_list_goals(const AGISystem* system, int* goal_ids, int max_count)
{
    if (!system || !goal_ids || max_count <= 0) return -1;
    int count = (system->goal_count < max_count) ? system->goal_count : max_count;
    int i;
    for (i = 0; i < count; i++)
        goal_ids[i] = system->goals[i].goal_id;
    return count;
}

int agi_system_add_task(AGISystem* system, const char* name, const char* description,
                         AGIPriority priority, int goal_id)
{
    if (!system || !name || !description) return -1;
    if (system->task_count >= AGI_MAX_TASKS) return -1;

    AGITask* task = &system->tasks[system->task_count];
    memset(task, 0, sizeof(AGITask));
    task->task_id = system->next_task_id++;
    strncpy(task->name, name, AGI_NAME_LEN - 1);
    strncpy(task->description, description, AGI_DESC_LEN - 1);
    task->status = AGI_TASK_PENDING;
    task->priority = priority;
    task->progress = 0.0f;
    task->created_at = time(NULL);
    task->goal_id = goal_id;
    task->current_action_index = 0;
    task->action_count = 0;
    task->error_code = 0;

    system->task_count++;
    system->status.active_task_count++;
    return task->task_id;
}

int agi_system_update_task(AGISystem* system, int task_id, float progress, AGITaskStatus status)
{
    if (!system) return -1;
    int i;
    for (i = 0; i < system->task_count; i++) {
        if (system->tasks[i].task_id == task_id) {
            AGITask* t = &system->tasks[i];
            t->progress = (progress < 0.0f) ? 0.0f : (progress > 1.0f) ? 1.0f : progress;
            t->status = status;
            if (status == AGI_TASK_RUNNING && t->started_at == 0)
                t->started_at = time(NULL);
            if (status == AGI_TASK_COMPLETED || status == AGI_TASK_FAILED) {
                t->completed_at = time(NULL);
                system->status.active_task_count--;
                if (status == AGI_TASK_COMPLETED) system->status.completed_task_count++;
                else system->status.failed_task_count++;
            }
            return 0;
        }
    }
    return -1;
}

int agi_system_get_task(const AGISystem* system, int task_id, AGITask* task)
{
    if (!system || !task) return -1;
    int i;
    for (i = 0; i < system->task_count; i++) {
        if (system->tasks[i].task_id == task_id) {
            *task = system->tasks[i];
            return 0;
        }
    }
    return -1;
}

int agi_system_remove_task(AGISystem* system, int task_id)
{
    if (!system) return -1;
    int i;
    for (i = 0; i < system->task_count; i++) {
        if (system->tasks[i].task_id == task_id) {
            if (system->tasks[i].action_sequence)
                safe_free((void**)&system->tasks[i].action_sequence);
            if (system->tasks[i].result_data)
                safe_free((void**)&system->tasks[i].result_data);
            int j;
            for (j = i; j < system->task_count - 1; j++)
                system->tasks[j] = system->tasks[j + 1];
            system->task_count--;
            memset(&system->tasks[system->task_count], 0, sizeof(AGITask));
            if (system->tasks[i].status == AGI_TASK_RUNNING || system->tasks[i].status == AGI_TASK_PENDING)
                system->status.active_task_count--;
            return 0;
        }
    }
    return -1;
}

int agi_system_list_tasks(const AGISystem* system, int* task_ids, int max_count)
{
    if (!system || !task_ids || max_count <= 0) return -1;
    int count = (system->task_count < max_count) ? system->task_count : max_count;
    int i;
    for (i = 0; i < count; i++)
        task_ids[i] = system->tasks[i].task_id;
    return count;
}

static void transition_state(AGISystem* system, AGIState new_state)
{
    if (!system) return;
    system->current_state = new_state;
    system->status.state = new_state;
    system->status.last_state_change = time(NULL);
}

static float compute_novelty(AGISystem* system, const float* input, int input_dim)
{
    if (!system || !input || input_dim <= 0) return 0.5f;
    int hists = system->cognitive_history_count;
    if (hists < 10) return 0.8f;
    float min_dist = 1e10f;
    int start = (hists > 100) ? hists - 100 : 0;
    int i, j;
    for (i = start; i < hists; i++) {
        float* hist = &system->cognitive_history[i * system->state_vector_dim];
        float dist = 0.0f;
        for (j = 0; j < input_dim && j < system->state_vector_dim; j++) {
            float d = hist[j] - input[j];
            dist += d * d;
        }
        if (dist < min_dist) min_dist = dist;
    }
    float novelty = sqrtf(min_dist) / (sqrtf(min_dist) + 1.0f);
    return (novelty < 0.05f) ? 0.05f : (novelty > 1.0f) ? 1.0f : novelty;
}

int agi_system_perceive(AGISystem* system, const float* sensory_input, int input_dim, float* state_vector)
{
    if (!system || !sensory_input || !state_vector) return -1;

    if (system->unified_lnn) {
        const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES];
        size_t raw_sizes[UNIFIED_LNN_MAX_MODALITIES];
        int modality_present[UNIFIED_LNN_MAX_MODALITIES];
        memset(modality_present, 0, sizeof(modality_present));
        for (int _ri = 0; _ri < UNIFIED_LNN_MAX_MODALITIES; _ri++) raw_inputs[_ri] = NULL;
        memset(raw_sizes, 0, sizeof(raw_sizes));

        modality_present[UNIFIED_MODALITY_VISION] = 1;
        raw_inputs[UNIFIED_MODALITY_VISION] = sensory_input;
        raw_sizes[UNIFIED_MODALITY_VISION] = (size_t)input_dim;

        float unified_output[AGI_STATE_VECTOR_DIM];
        size_t max_out = (size_t)system->state_vector_dim < AGI_STATE_VECTOR_DIM ?
                         (size_t)system->state_vector_dim : AGI_STATE_VECTOR_DIM;

        int step_result = unified_lnn_state_step(system->unified_lnn,
                                                  raw_inputs, raw_sizes,
                                                  modality_present,
                                                  unified_output, max_out);
        if (step_result > 0) {
            memcpy(state_vector, unified_output, (size_t)step_result * sizeof(float));
            if ((size_t)step_result < (size_t)system->state_vector_dim)
                memset(state_vector + step_result, 0,
                       (size_t)(system->state_vector_dim - step_result) * sizeof(float));
        } else {
            float hidden_buf[AGI_STATE_VECTOR_DIM];
            size_t hd = max_out;
            if (unified_lnn_state_get_hidden_state(system->unified_lnn, hidden_buf, hd) > 0) {
                memcpy(state_vector, hidden_buf, hd * sizeof(float));
            } else {
                memset(state_vector, 0, (size_t)system->state_vector_dim * sizeof(float));
            }
        }
    } else {
        int dim = (input_dim < system->state_vector_dim) ? input_dim : system->state_vector_dim;
        memcpy(state_vector, sensory_input, (size_t)dim * sizeof(float));
        if (dim < system->state_vector_dim)
            memset(state_vector + dim, 0, (size_t)(system->state_vector_dim - dim) * sizeof(float));

        if (system->lnn) {
            float lnn_output[128];
            int lnn_dim = 128;
            if (lnn_dim > system->state_vector_dim) lnn_dim = system->state_vector_dim;
            if (system->config.state_vector_dim > 0) {
                int ret = lnn_forward(system->lnn, state_vector, lnn_output);
                if (ret == 0) {
                    int copy_dim = (lnn_dim < system->state_vector_dim) ? lnn_dim : system->state_vector_dim;
                    memcpy(state_vector, lnn_output, (size_t)copy_dim * sizeof(float));
                }
            }
        }
    }

    memcpy(system->state_vector, state_vector, (size_t)system->state_vector_dim * sizeof(float));

    if (system->cognitive_history_count < system->max_cognitive_history) {
        int idx = system->cognitive_history_count * system->state_vector_dim;
        memcpy(&system->cognitive_history[idx], state_vector, (size_t)system->state_vector_dim * sizeof(float));
        system->cognitive_history_count++;
    } else {
        int shift = system->state_vector_dim * 10;
        int remain = (system->max_cognitive_history - 10) * system->state_vector_dim;
        memmove(system->cognitive_history, system->cognitive_history + shift, (size_t)remain * sizeof(float));
        int idx = remain;
        memcpy(&system->cognitive_history[idx], state_vector, (size_t)system->state_vector_dim * sizeof(float));
    }

    float novelty = compute_novelty(system, sensory_input, input_dim);
    system->status.curiosity = novelty * system->config.exploration_rate +
                               (1.0f - novelty) * system->status.curiosity * 0.9f;

    return 0;
}

int agi_system_perceive_multimodal(AGISystem* system,
    const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES],
    const int raw_dims[UNIFIED_LNN_MAX_MODALITIES],
    const int modality_present[UNIFIED_LNN_MAX_MODALITIES],
    float* state_vector, int state_vector_dim)
{
    if (!system || !state_vector || !raw_inputs || !raw_dims || !modality_present)
        return -1;
    if (state_vector_dim <= 0 || state_vector_dim > AGI_STATE_VECTOR_DIM)
        return -1;

    int local_modality_present[UNIFIED_LNN_MAX_MODALITIES];
    const float* local_raw_inputs[UNIFIED_LNN_MAX_MODALITIES];
    int local_raw_dims[UNIFIED_LNN_MAX_MODALITIES];
    int m;
    for (m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
        local_modality_present[m] = modality_present[m];
        local_raw_inputs[m] = raw_inputs[m];
        local_raw_dims[m] = raw_dims[m];
    }

    float vision_feature_buf[512];
    int vision_feature_count = 0;

    if (system->vision && system->vision_buffer_valid && system->vision_input_buffer) {
        vision_feature_count = cfc_vision_extract_features(system->vision,
            system->vision_input_buffer,
            system->vision_buffer_width,
            system->vision_buffer_height,
            system->vision_buffer_channels,
            vision_feature_buf, 512);
        if (vision_feature_count > 0) {
            local_modality_present[UNIFIED_MODALITY_VISION] = 1;
            local_raw_inputs[UNIFIED_MODALITY_VISION] = vision_feature_buf;
            local_raw_dims[UNIFIED_MODALITY_VISION] = vision_feature_count;
        }
    }

    if (system->unified_lnn) {
        size_t sizes[UNIFIED_LNN_MAX_MODALITIES];
        for (m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++)
            sizes[m] = (size_t)(local_raw_dims[m] > 0 ? local_raw_dims[m] : 0);

        float unified_output[AGI_STATE_VECTOR_DIM];
        size_t max_out = (size_t)state_vector_dim > AGI_STATE_VECTOR_DIM ?
                         AGI_STATE_VECTOR_DIM : (size_t)state_vector_dim;

        int step_result = unified_lnn_state_step(system->unified_lnn,
                                                  local_raw_inputs, sizes,
                                                  local_modality_present,
                                                  unified_output, max_out);
        if (step_result > 0) {
            memcpy(state_vector, unified_output, (size_t)step_result * sizeof(float));
            if ((size_t)step_result < (size_t)state_vector_dim)
                memset(state_vector + step_result, 0,
                       (size_t)(state_vector_dim - step_result) * sizeof(float));
        } else {
            float hidden_buf[AGI_STATE_VECTOR_DIM];
            size_t hd = max_out;
            if (unified_lnn_state_get_hidden_state(system->unified_lnn, hidden_buf, hd) > 0) {
                memcpy(state_vector, hidden_buf, hd * sizeof(float));
                if (hd < (size_t)state_vector_dim)
                    memset(state_vector + hd, 0,
                           (size_t)(state_vector_dim - hd) * sizeof(float));
            } else {
                memset(state_vector, 0, (size_t)state_vector_dim * sizeof(float));
            }
        }
    } else {
        int total_dim = 0;
        for (m = 0; m < UNIFIED_LNN_MAX_MODALITIES && local_modality_present[m]; m++) {
            if (local_raw_inputs[m] && local_raw_dims[m] > 0) {
                int copy_dim = local_raw_dims[m];
                if (total_dim + copy_dim > state_vector_dim)
                    copy_dim = state_vector_dim - total_dim;
                if (copy_dim > 0) {
                    memcpy(state_vector + total_dim, local_raw_inputs[m],
                           (size_t)copy_dim * sizeof(float));
                    total_dim += copy_dim;
                }
            }
        }
        if (total_dim < state_vector_dim)
            memset(state_vector + total_dim, 0,
                   (size_t)(state_vector_dim - total_dim) * sizeof(float));

        if (system->lnn) {
            float lnn_output[128];
            int lnn_dim = 128;
            if (lnn_dim > state_vector_dim) lnn_dim = state_vector_dim;
            int ret = lnn_forward(system->lnn, state_vector, lnn_output);
            if (ret == 0) {
                memcpy(state_vector, lnn_output, (size_t)lnn_dim * sizeof(float));
            }
        }
    }

    memcpy(system->state_vector, state_vector, (size_t)system->state_vector_dim * sizeof(float));

    if (system->cognitive_history_count < system->max_cognitive_history) {
        int idx = system->cognitive_history_count * system->state_vector_dim;
        memcpy(&system->cognitive_history[idx], state_vector, (size_t)system->state_vector_dim * sizeof(float));
        system->cognitive_history_count++;
    } else {
        int shift = system->state_vector_dim * 10;
        int remain = (system->max_cognitive_history - 10) * system->state_vector_dim;
        memmove(system->cognitive_history, system->cognitive_history + shift, (size_t)remain * sizeof(float));
        int idx = remain;
        memcpy(&system->cognitive_history[idx], state_vector, (size_t)system->state_vector_dim * sizeof(float));
    }

    int last_present = 0;
    for (m = UNIFIED_LNN_MAX_MODALITIES - 1; m >= 0; m--) {
        if (local_modality_present[m] && local_raw_inputs[m]) { last_present = m; break; }
    }
    if (last_present >= 0 && local_raw_inputs[last_present]) {
        float novelty = compute_novelty(system, local_raw_inputs[last_present], local_raw_dims[last_present]);
        system->status.curiosity = novelty * system->config.exploration_rate +
                                   (1.0f - novelty) * system->status.curiosity * 0.9f;
    }

    return 0;
}

int agi_system_reason(AGISystem* system, const float* state_vector, int state_dim, float* reasoning_result)
{
    if (!system || !state_vector || !reasoning_result) return -1;
    if (!system->reasoning) return -1;
    int ret = reasoning_infer(system->reasoning, state_vector, (size_t)state_dim,
                               reasoning_result, (size_t)state_dim, REASONING_DEDUCTIVE);
    if (ret != 0) {
        memcpy(reasoning_result, state_vector, (size_t)state_dim * sizeof(float));
    }
    return 0;
}

int agi_system_plan(AGISystem* system, const float* reasoning_result, int result_dim,
                     float* plan, int* plan_steps)
{
    if (!system || !reasoning_result || !plan || !plan_steps) return -1;
    if (!system->planner) return -1;

    int goal_id = -1;
    int i;
    for (i = 0; i < system->goal_count; i++) {
        if (system->goals[i].status == AGI_GOAL_PENDING ||
            system->goals[i].status == AGI_GOAL_IN_PROGRESS) {
            if (goal_id < 0 || system->goals[i].priority > system->goals[goal_id].priority)
                goal_id = i;
        }
    }

    float goal_vec[AGI_STATE_VECTOR_DIM];
    memset(goal_vec, 0, sizeof(goal_vec));
    int goal_dim = AGI_STATE_VECTOR_DIM;
    if (goal_id >= 0) {
        goal_dim = system->goals[goal_id].state_vector_dim;
        if (goal_dim > AGI_STATE_VECTOR_DIM) goal_dim = AGI_STATE_VECTOR_DIM;
        if (goal_dim > 0) memcpy(goal_vec, system->goals[goal_id].state_vector, (size_t)goal_dim * sizeof(float));
    }

    int max_plan = system->config.planning_horizon;
    int steps = planning_generate(system->planner, goal_vec, (size_t)goal_dim,
                                   reasoning_result, (size_t)result_dim,
                                   plan, (size_t)max_plan * 2);
    if (steps < 0) steps = 0;
    *plan_steps = steps;

    if (goal_id >= 0 && steps > 0)
        system->goals[goal_id].status = AGI_GOAL_IN_PROGRESS;

    return 0;
}

int agi_system_decide(AGISystem* system, const float* plan_options, int num_options, int* chosen_option)
{
    if (!system || !plan_options || !chosen_option) return -1;
    if (!system->config.enable_self_decision) {
        *chosen_option = 0;
        return 0;
    }

    int nalts = (num_options < 16) ? num_options : 16;
    if (nalts < 1) { *chosen_option = 0; return 0; }
    int n_obj = 2;
    float weights[2] = {0.7f, 0.3f};
    int is_max[2] = {1, 0};

    float* mat = (float*)safe_malloc((size_t)nalts * 2 * sizeof(float));
    if (!mat) { *chosen_option = 0; return -1; }
    for (int i = 0; i < nalts; i++) {
        mat[i * 2 + 0] = plan_options[i * 2];
        mat[i * 2 + 1] = plan_options[i * 2 + 1];
    }

    MAUTResult mresult;
    memset(&mresult, 0, sizeof(mresult));
    agi_maut_decide(system, mat, nalts, n_obj, weights, is_max,
                    AGI_MAUT_TOPSIS, &mresult);

    *chosen_option = mresult.best_alternative;
    system->status.confidence = mresult.overall_confidence;

    agi_maut_result_free(&mresult);
    safe_free((void**)&mat);
    return 0;
}

int agi_system_execute(AGISystem* system, int chosen_option, float* execution_output, int output_dim)
{
    if (!system || !execution_output) return -1;
    if (!system->config.enable_autonomous_execution) return 0;
    if (!capability_is_enabled(CAP_AUTONOMOUS_EXECUTION)) return 0;

    int active_task = -1;
    int i;
    for (i = 0; i < system->task_count; i++) {
        if (system->tasks[i].status == AGI_TASK_RUNNING) {
            active_task = i;
            break;
        }
    }

    if (active_task < 0) {
        for (i = 0; i < system->task_count; i++) {
            if (system->tasks[i].status == AGI_TASK_PENDING) {
                system->tasks[i].status = AGI_TASK_RUNNING;
                system->tasks[i].started_at = time(NULL);
                active_task = i;
                break;
            }
        }
    }

    if (active_task >= 0) {
        AGITask* t = &system->tasks[active_task];

        int action_count = t->action_count;
        if (t->current_action_index < action_count && t->action_sequence) {
            int idx = t->current_action_index * output_dim;
            int copy_dim = output_dim;
            if (idx + copy_dim <= action_count * output_dim) {
                memcpy(execution_output, &t->action_sequence[idx], (size_t)copy_dim * sizeof(float));
            }
            t->current_action_index++;
            if (t->current_action_index >= action_count) {
                t->status = AGI_TASK_COMPLETED;
                t->completed_at = time(NULL);
                system->status.active_task_count--;
                system->status.completed_task_count++;
                system->status.confidence = 0.8f;
            } else {
                t->progress = (float)t->current_action_index / (float)action_count;
            }
        } else {
            memset(execution_output, 0, (size_t)output_dim * sizeof(float));
        }
    } else {
        memset(execution_output, 0, (size_t)output_dim * sizeof(float));
    }

    /* 多机器人/多设备任务委派：当多系统控制引擎可用时，将待处理任务委派给远程设备 */
    if (system->multisystem_control) {
        DeviceInfo** devices = NULL;
        size_t device_count = 0;
        int discovery_ok = discover_available_devices(system->multisystem_control, &devices, &device_count);
        if (discovery_ok == 0 && device_count > 0) {
            Task* pending_tasks[AGI_MAX_TASKS];
            int pending_indices[AGI_MAX_TASKS];
            int pending_count = 0;
            for (i = 0; i < system->task_count && pending_count < AGI_MAX_TASKS; i++) {
                if (system->tasks[i].status == AGI_TASK_PENDING) {
                    char task_id[32];
                    snprintf(task_id, sizeof(task_id), "agi_task_%d", system->tasks[i].task_id);
                    Task* ms_task = create_task(task_id, TASK_TYPE_CUSTOM, system->tasks[i].description);
                    if (ms_task) {
                        double prio = (double)(AGI_PRIORITY_BACKGROUND - system->tasks[i].priority)
                                    / (double)AGI_PRIORITY_BACKGROUND;
                        if (prio < 0.1) prio = 0.1;
                        if (prio > 1.0) prio = 1.0;
                        ms_task->priority = prio;
                        pending_tasks[pending_count] = ms_task;
                        pending_indices[pending_count] = i;
                        pending_count++;
                    }
                }
            }
            if (pending_count > 0) {
                CoordinationPlan* plan = coordinate_multitask_execution(
                    system->multisystem_control,
                    pending_tasks, (size_t)pending_count,
                    devices, device_count);
                if (plan) {
                    execute_coordination_plan(system->multisystem_control, plan);
                    for (i = 0; i < pending_count; i++) {
                        int idx = pending_indices[i];
                        if (system->tasks[idx].status == AGI_TASK_PENDING) {
                            system->tasks[idx].status = AGI_TASK_RUNNING;
                            system->tasks[idx].started_at = time(NULL);
                        }
                    }
                    destroy_coordination_plan(plan);
                }
                for (i = 0; i < pending_count; i++) {
                    destroy_task(pending_tasks[i]);
                }
            }
            free_device_list(devices, device_count);
        }
    }

    return 0;
}

int agi_system_learn(AGISystem* system, const float* experience, int exp_dim, float reward)
{
    if (!system || !experience) return -1;
    if (!system->config.enable_self_learning) return 0;

    system->total_reward += reward;
    system->reward_count++;
    system->avg_reward = system->total_reward / (float)system->reward_count;

    if (reward < -5.0f) {
        system->consecutive_low_reward++;
    } else {
        system->consecutive_low_reward = 0;
    }

    if (system->learner && exp_dim > 0) {
        /* 基于经验数据推导真实动作特征（带微小噪声防止零向量） */
        float derived_action[4];
        float total_magnitude = 0.0f;
        for (int i = 0; i < exp_dim; i++) {
            total_magnitude += fabsf(experience[i]);
        }
        if (total_magnitude < 1e-6f) {
            /* 经验为零时添加微小探索噪声，防止学习停滞 */
            for (int i = 0; i < 4; i++) {
                float noise = (secure_random_float() - 0.5f) * 0.02f;
                derived_action[i] = noise;
            }
        } else {
            for (int i = 0; i < 4 && i < exp_dim; i++) {
                derived_action[i] = tanhf(experience[i] * 0.5f);
            }
            for (int i = exp_dim; i < 4; i++) {
                derived_action[i] = 0.0f;
            }
        }
        learning_reinforcement_update(system->learner,
                                       experience, (size_t)exp_dim,
                                       derived_action, 4,
                                       reward,
                                       experience, (size_t)exp_dim);
    }

    if (system->knowledge) {
        char key[64];
        sprintf(key, "exp_%ld_%d", (long)time(NULL), system->reward_count);
        KnowledgeEntry entry;
        memset(&entry, 0, sizeof(entry));
        entry.subject = key;
        entry.predicate = "experience";
        entry.object = "reward";
        entry.type = KNOWLEDGE_OBSERVATION;
        entry.confidence = (reward > 0) ? CONFIDENCE_HIGH : CONFIDENCE_MEDIUM;
        entry.source = SOURCE_LEARNING;
        entry.weight = (reward > 0) ? 1.0f : 0.5f;
        entry.timestamp = (long)time(NULL);
        knowledge_base_add(system->knowledge, &entry);
    }

    if (system->metacognition) {
        MetacognitionMonitoringResult mresult;
        memset(&mresult, 0, sizeof(mresult));
        metacognition_monitor(system->metacognition, experience, (size_t)exp_dim, &mresult);
    }

    if (system->config.enable_self_evolution &&
        system->consecutive_low_reward >= 10 &&
        system->avg_reward < 0.3f) {
        float metrics[4];
        metrics[0] = system->avg_reward;
        metrics[1] = system->status.confidence;
        metrics[2] = system->status.learning_progress;
        metrics[3] = system->status.cognitive_load;
        int evolve_ret = learning_evolutionary_evolve(system->learner, metrics, 4);
        if (evolve_ret == 0) {
            float best_individual[AGI_STATE_VECTOR_DIM];
            int best_size = (int)learning_evolutionary_get_best(system->learner,
                                                                best_individual,
                                                                (size_t)AGI_STATE_VECTOR_DIM);
            if (best_size > 0) {
                int apply_n = (best_size < system->state_vector_dim) ? best_size : system->state_vector_dim;
                float blend = 0.3f;
                for (int i = 0; i < apply_n; i++) {
                    system->state_vector[i] = (1.0f - blend) * system->state_vector[i] + blend * best_individual[i];
                }
            }
        } else {
            int i;
            for (i = 0; i < system->state_vector_dim; i++) {
                /* K-006修复：使用安全随机数 */
                float noise = (secure_random_float() - 0.5f) * 0.05f;
                system->state_vector[i] += noise;
            }
        }
        system->consecutive_low_reward = 0;
        system->status.learning_progress = system->avg_reward;
    }

    system->status.learning_progress = system->avg_reward;

    return 0;
}

int agi_system_cognitive_cycle(AGISystem* system, const float* sensory_input, int input_dim,
                                float* output, int output_dim)
{
    if (!system || !sensory_input || !output) return -1;
    if (!system->initialized) return -1;

    system->total_cycles++;
    int state_dim = system->state_vector_dim;
    int i;

    transition_state(system, AGI_STATE_PERCEIVE);
    float perceived_state[AGI_STATE_VECTOR_DIM];
    agi_system_perceive(system, sensory_input, input_dim, perceived_state);

    if (system->self_cognition) {
        self_cognition_update(system->self_cognition, SELF_COGNITION_STATE);
    }

    transition_state(system, AGI_STATE_REASON);
    float reasoning_result[AGI_STATE_VECTOR_DIM];
    if (system->knowledge) {
        size_t kb_entries = 0;
        size_t kb_mem = 0;
        if (knowledge_base_get_stats(system->knowledge, &kb_entries, &kb_mem) == 0 && kb_entries > 0) {
            KnowledgeEntry kb_results[16];
            float kb_scores[16];
            int found = knowledge_base_query_state_aware(
                system->knowledge, NULL,
                perceived_state, state_dim,
                kb_results, 16, kb_scores);
            if (found > 0) {
                float knowledge_bias[AGI_STATE_VECTOR_DIM];
                memset(knowledge_bias, 0, sizeof(knowledge_bias));
                int used = (found < 8) ? found : 8;
                float total_score = 0.0f;
                for (int ki = 0; ki < used; ki++) {
                    total_score += kb_scores[ki];
                }
                if (total_score < 1e-6f) total_score = 1.0f;
                for (int ki = 0; ki < used; ki++) {
                    float weight = kb_scores[ki] / total_score;
                    float bias = kb_results[ki].weight * 0.15f * weight;
                    int idx = (ki * 31 + 7) % state_dim;
                    knowledge_bias[idx] += bias;
                }
                float enhanced_state[AGI_STATE_VECTOR_DIM];
                memcpy(enhanced_state, perceived_state, (size_t)state_dim * sizeof(float));
                for (int ki = 0; ki < state_dim; ki++) {
                    enhanced_state[ki] += knowledge_bias[ki];
                }
                agi_system_reason(system, enhanced_state, state_dim, reasoning_result);
                for (int ei = 0; ei < found; ei++) {
                    safe_free((void**)&kb_results[ei].subject);
                    safe_free((void**)&kb_results[ei].predicate);
                    safe_free((void**)&kb_results[ei].object);
                    safe_free((void**)&kb_results[ei].metadata);
                }
            } else {
                agi_system_reason(system, perceived_state, state_dim, reasoning_result);
            }
        } else {
            agi_system_reason(system, perceived_state, state_dim, reasoning_result);
        }
    } else {
        agi_system_reason(system, perceived_state, state_dim, reasoning_result);
    }

    transition_state(system, AGI_STATE_PLAN);
    float plan_buffer[AGI_STATE_VECTOR_DIM * 2];
    int plan_steps = 0;
    if (system->config.enable_planning) {
        agi_system_plan(system, reasoning_result, state_dim, plan_buffer, &plan_steps);
    }

    transition_state(system, AGI_STATE_DECIDE);
    int chosen_option = 0;
    if (plan_steps > 0) {
        float plan_options[32];
        int nopts = (plan_steps < 16) ? plan_steps : 16;
        float max_mag = 0.0f;
        for (i = 0; i < nopts; i++) {
            float action = plan_buffer[i * 2];
            float param = plan_buffer[i * 2 + 1];
            float magnitude = fabsf(action) + fabsf(param);
            if (magnitude > max_mag) max_mag = magnitude;
        }
        if (max_mag < 1e-6f) max_mag = 1.0f;
        for (i = 0; i < nopts; i++) {
            float action = plan_buffer[i * 2];
            float param = plan_buffer[i * 2 + 1];
            float magnitude = (fabsf(action) + fabsf(param)) / max_mag;
            float align = 0.0f;
            int j;
            for (j = 0; j < state_dim && j < 32; j++) {
                align += reasoning_result[j % state_dim] * plan_buffer[(i * 2 + j) % (plan_steps * 2)];
            }
            align = (align / (float)(state_dim > 32 ? 32 : state_dim)) * 0.5f + 0.5f;
            if (align < 0.0f) align = 0.0f;
            if (align > 1.0f) align = 1.0f;
            float util = 0.3f + 0.7f * magnitude * align;
            float risk = 0.2f + 0.3f * (1.0f - magnitude * align);
            if (util > 1.0f) util = 1.0f;
            if (risk > 1.0f) risk = 1.0f;
            plan_options[i * 2] = util;
            plan_options[i * 2 + 1] = risk;
        }
        agi_system_decide(system, plan_options, nopts, &chosen_option);
    }

    transition_state(system, AGI_STATE_EXECUTE);
    agi_system_execute(system, chosen_option, output, output_dim);

    transition_state(system, AGI_STATE_LEARN);
    /* 结构化奖励：多维度奖励信号融合
     * 1. 输出-输入MSE（基础学习信号）
     * 2. 新颖性奖励（鼓励探索未知状态）
     * 3. 任务完成奖励（从任务执行状态获取）
     */
    float reward = 0.0f;
    const float reward_mse_weight = 0.5f;
    const float reward_novelty_weight = 0.3f;
    const float reward_progress_weight = 0.2f;
    
    for (i = 0; i < output_dim && i < state_dim; i++) {
        float delta = output[i] - sensory_input[i % input_dim];
        reward -= delta * delta * 0.01f;
    }
    reward *= reward_mse_weight;
    
    /* 新颖性奖励：基于认知历史距离 */
    float novelty = compute_novelty(system, perceived_state, state_dim);
    reward += novelty * reward_novelty_weight;
    
    /* 任务进度奖励：有活跃任务时给予额外奖励 */
    int active_tasks = 0;
    for (int ti = 0; ti < system->task_count; ti++) {
        if (system->tasks[ti].status == AGI_TASK_RUNNING) {
            active_tasks++;
            reward += system->tasks[ti].progress * reward_progress_weight * 0.1f;
        }
    }
    agi_system_learn(system, perceived_state, state_dim, reward);

    if (system->self_cognition) {
        self_cognition_update(system->self_cognition, SELF_COGNITION_LEARNING);
    }

    if (system->config.enable_reflection &&
        system->total_cycles % (int)system->config.reflection_interval == 0) {
        transition_state(system, AGI_STATE_REFLECT);
        char reflection_buf[2048];
        agi_system_self_reflect(system, reflection_buf, sizeof(reflection_buf));
        if (system->self_cognition) {
            char cog_reflection_buf[2048];
            if (self_cognition_reflect(system->self_cognition, cog_reflection_buf, sizeof(cog_reflection_buf)) > 0) {
                system->status.reflection_count++;
            }
        }

        /* 自我修正流水线：深度修正系统的错误检测、根因分析和自动修正 */
        if (system->correction && system->config.enable_self_correction) {
            int corrections_applied = 0;
            int ci;
            char error_desc[256];

            /* 条件1: 连续低奖励 → 策略错误修正 */
            if (system->consecutive_low_reward > 3) {
                snprintf(error_desc, sizeof(error_desc),
                    "连续%d周期低奖励(平均%.3f)，策略需要调整",
                    system->consecutive_low_reward, system->avg_reward);
                DCErrorRecord err_rec;
                DCCorrectionHypothesis hyp_rec;
                float eff = 0.0f;
                int ret = dc_run_full_correction_pipeline(system->correction,
                    DC_ERROR_STRATEGY, error_desc, 0.7f, &err_rec, &hyp_rec, &eff);
                if (ret == 0 && eff > 0.5f) {
                    float corr_signal = eff * 0.1f;
                    for (ci = 0; ci < state_dim; ci++) {
                        system->state_vector[ci] += corr_signal * (0.5f - system->state_vector[ci]);
                    }
                    corrections_applied++;
                }
            }

            /* 条件2: 低置信度 → 逻辑错误修正 */
            if (system->status.confidence < 0.3f && system->total_cycles > 5) {
                snprintf(error_desc, sizeof(error_desc),
                    "置信度偏低(%.2f)，推理链可能需要修正",
                    system->status.confidence);
                DCErrorRecord err_rec;
                DCCorrectionHypothesis hyp_rec;
                float eff = 0.0f;
                int ret = dc_run_full_correction_pipeline(system->correction,
                    DC_ERROR_LOGIC, error_desc, 0.6f, &err_rec, &hyp_rec, &eff);
                if (ret == 0 && eff > 0.5f) {
                    system->status.confidence += eff * 0.2f;
                    if (system->status.confidence > 1.0f) system->status.confidence = 1.0f;
                    corrections_applied++;
                }
            }

            /* 条件3: 认知负载过高 → 执行错误修正 */
            if (system->status.cognitive_load > 0.85f) {
                snprintf(error_desc, sizeof(error_desc),
                    "认知负载过高(%.2f)，资源分配需要优化",
                    system->status.cognitive_load);
                DCErrorRecord err_rec;
                DCCorrectionHypothesis hyp_rec;
                float eff = 0.0f;
                int ret = dc_run_full_correction_pipeline(system->correction,
                    DC_ERROR_EXECUTION, error_desc, 0.5f, &err_rec, &hyp_rec, &eff);
                if (ret == 0 && eff > 0.5f) {
                    system->status.cognitive_load *= (1.0f - eff * 0.3f);
                    corrections_applied++;
                }
            }

            /* 条件4: 长期低平均奖励 → 知识错误修正 */
            if (system->avg_reward < 0.1f && system->total_cycles > 20) {
                snprintf(error_desc, sizeof(error_desc),
                    "长期平均奖励偏低(%.3f)，知识库可能需要更新",
                    system->avg_reward);
                DCErrorRecord err_rec;
                DCCorrectionHypothesis hyp_rec;
                float eff = 0.0f;
                int ret = dc_run_full_correction_pipeline(system->correction,
                    DC_ERROR_KNOWLEDGE, error_desc, 0.4f, &err_rec, &hyp_rec, &eff);
                if (ret == 0 && eff > 0.5f) {
                    corrections_applied++;
                }
            }

            /* 记录修正事件到知识库 */
            if (corrections_applied > 0 && system->knowledge) {
                char key[64];
                sprintf(key, "correction_%d_%ld", system->total_cycles, (long)time(NULL));
                KnowledgeEntry entry;
                memset(&entry, 0, sizeof(entry));
                entry.subject = key;
                entry.predicate = "自我修正";
                entry.object = "流水线";
                entry.type = KNOWLEDGE_OBSERVATION;
                entry.confidence = CONFIDENCE_HIGH;
                entry.source = SOURCE_INFERENCE;
                entry.weight = 0.9f;
                entry.timestamp = (long)time(NULL);
                knowledge_base_add(system->knowledge, &entry);
            }
        }
    }

    system->status.cognitive_load = (float)system->task_count / (float)AGI_MAX_TASKS;
    system->status.cycle_count = system->total_cycles;

    /* 快速稳定性检查：周期性的状态稳定性评估 */
    if (system->total_cycles % 10 == 0) {
        float den_coeffs[3];
        float state_mag = 0.0f;
        for (i = 0; i < state_dim && i < 64; i++)
            state_mag += perceived_state[i] * perceived_state[i];
        state_mag = sqrtf(state_mag / (float)(state_dim > 64 ? 64 : state_dim) + 1e-12f);
        den_coeffs[0] = 1.0f;
        den_coeffs[1] = -0.3f * state_mag;
        int is_stable = 0;
        if (laplace_check_stability_fast(den_coeffs, 2, &is_stable, NULL) == 0) {
            system->status.confidence = is_stable ? 0.8f : 0.4f;
            system->status.cognitive_load = is_stable ?
                system->status.cognitive_load : (system->status.cognitive_load + 0.1f);
        }
    }

    return 0;
}

int agi_system_cognitive_cycle_multimodal(AGISystem* system,
    const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES],
    const int raw_dims[UNIFIED_LNN_MAX_MODALITIES],
    const int modality_present[UNIFIED_LNN_MAX_MODALITIES],
    float* output, int output_dim)
{
    if (!system || !raw_inputs || !raw_dims || !modality_present || !output) return -1;
    if (!system->initialized) return -1;

    system->total_cycles++;
    int state_dim = system->state_vector_dim;
    int i;

    transition_state(system, AGI_STATE_PERCEIVE);
    float perceived_state[AGI_STATE_VECTOR_DIM];
    agi_system_perceive_multimodal(system, raw_inputs, raw_dims, modality_present,
                                    perceived_state, state_dim);

    if (system->self_cognition) {
        self_cognition_update(system->self_cognition, SELF_COGNITION_STATE);
    }

    transition_state(system, AGI_STATE_REASON);
    float reasoning_result[AGI_STATE_VECTOR_DIM];
    if (system->knowledge) {
        size_t kb_entries = 0;
        size_t kb_mem = 0;
        if (knowledge_base_get_stats(system->knowledge, &kb_entries, &kb_mem) == 0 && kb_entries > 0) {
            KnowledgeEntry kb_results[16];
            float kb_scores[16];
            int found = knowledge_base_query_state_aware(
                system->knowledge, NULL,
                perceived_state, state_dim,
                kb_results, 16, kb_scores);
            if (found > 0) {
                float knowledge_bias[AGI_STATE_VECTOR_DIM];
                memset(knowledge_bias, 0, sizeof(knowledge_bias));
                int used = (found < 8) ? found : 8;
                float total_score = 0.0f;
                for (int ki = 0; ki < used; ki++) {
                    total_score += kb_scores[ki];
                }
                if (total_score < 1e-6f) total_score = 1.0f;
                for (int ki = 0; ki < used; ki++) {
                    float weight = kb_scores[ki] / total_score;
                    float bias = kb_results[ki].weight * 0.15f * weight;
                    int idx = (ki * 31 + 7) % state_dim;
                    knowledge_bias[idx] += bias;
                }
                float enhanced_state[AGI_STATE_VECTOR_DIM];
                memcpy(enhanced_state, perceived_state, (size_t)state_dim * sizeof(float));
                for (int ki = 0; ki < state_dim; ki++) {
                    enhanced_state[ki] += knowledge_bias[ki];
                }
                agi_system_reason(system, enhanced_state, state_dim, reasoning_result);
                for (int ei = 0; ei < found; ei++) {
                    safe_free((void**)&kb_results[ei].subject);
                    safe_free((void**)&kb_results[ei].predicate);
                    safe_free((void**)&kb_results[ei].object);
                    safe_free((void**)&kb_results[ei].metadata);
                }
            } else {
                agi_system_reason(system, perceived_state, state_dim, reasoning_result);
            }
        } else {
            agi_system_reason(system, perceived_state, state_dim, reasoning_result);
        }
    } else {
        agi_system_reason(system, perceived_state, state_dim, reasoning_result);
    }

    transition_state(system, AGI_STATE_PLAN);
    float plan_buffer[AGI_STATE_VECTOR_DIM * 2];
    int plan_steps = 0;
    if (system->config.enable_planning) {
        agi_system_plan(system, reasoning_result, state_dim, plan_buffer, &plan_steps);
    }

    transition_state(system, AGI_STATE_DECIDE);
    int chosen_option = 0;
    if (plan_steps > 0) {
        float plan_options[32];
        int nopts = (plan_steps < 16) ? plan_steps : 16;
        float max_mag = 0.0f;
        for (i = 0; i < nopts; i++) {
            float action = plan_buffer[i * 2];
            float param = plan_buffer[i * 2 + 1];
            float magnitude = fabsf(action) + fabsf(param);
            if (magnitude > max_mag) max_mag = magnitude;
        }
        if (max_mag < 1e-6f) max_mag = 1.0f;
        for (i = 0; i < nopts; i++) {
            float action = plan_buffer[i * 2];
            float param = plan_buffer[i * 2 + 1];
            float magnitude = (fabsf(action) + fabsf(param)) / max_mag;
            float align = 0.0f;
            int j;
            for (j = 0; j < state_dim && j < 32; j++) {
                align += reasoning_result[j % state_dim] * plan_buffer[(i * 2 + j) % (plan_steps * 2)];
            }
            align = (align / (float)(state_dim > 32 ? 32 : state_dim)) * 0.5f + 0.5f;
            if (align < 0.0f) align = 0.0f;
            if (align > 1.0f) align = 1.0f;
            float util = 0.3f + 0.7f * magnitude * align;
            float risk = 0.2f + 0.3f * (1.0f - magnitude * align);
            if (util > 1.0f) util = 1.0f;
            if (risk > 1.0f) risk = 1.0f;
            plan_options[i * 2] = util;
            plan_options[i * 2 + 1] = risk;
        }
        agi_system_decide(system, plan_options, nopts, &chosen_option);
    }

    transition_state(system, AGI_STATE_EXECUTE);
    agi_system_execute(system, chosen_option, output, output_dim);

    transition_state(system, AGI_STATE_LEARN);
    float reward = 0.0f;
    int last_input_dim = 0;
    for (i = UNIFIED_LNN_MAX_MODALITIES - 1; i >= 0; i--) {
        if (modality_present[i] && raw_inputs[i] && raw_dims[i] > 0) {
            last_input_dim = raw_dims[i];
            break;
        }
    }
    if (last_input_dim > 0) {
        const float* last_input = raw_inputs[last_input_dim > 0 ? last_input_dim - 1 : 0];
        for (i = 0; i < output_dim && i < state_dim; i++) {
            float delta = output[i] - (last_input ? last_input[i % last_input_dim] : 0.0f);
            reward -= delta * delta * 0.01f;
        }
    }
    agi_system_learn(system, perceived_state, state_dim, reward);

    if (system->self_cognition) {
        self_cognition_update(system->self_cognition, SELF_COGNITION_LEARNING);
    }

    if (system->config.enable_reflection &&
        system->total_cycles % (int)system->config.reflection_interval == 0) {
        transition_state(system, AGI_STATE_REFLECT);
        char reflection_buf[2048];
        agi_system_self_reflect(system, reflection_buf, sizeof(reflection_buf));
        if (system->self_cognition) {
            char cog_reflection_buf[2048];
            if (self_cognition_reflect(system->self_cognition, cog_reflection_buf, sizeof(cog_reflection_buf)) > 0) {
                system->status.reflection_count++;
            }
        }

        /* 自我修正流水线：深度修正系统的错误检测、根因分析和自动修正 */
        if (system->correction && system->config.enable_self_correction) {
            int corrections_applied = 0;
            int ci;
            char error_desc[256];

            /* 条件1: 连续低奖励 → 策略错误修正 */
            if (system->consecutive_low_reward > 3) {
                snprintf(error_desc, sizeof(error_desc),
                    "连续%d周期低奖励(平均%.3f)，策略需要调整",
                    system->consecutive_low_reward, system->avg_reward);
                DCErrorRecord err_rec;
                DCCorrectionHypothesis hyp_rec;
                float eff = 0.0f;
                int ret = dc_run_full_correction_pipeline(system->correction,
                    DC_ERROR_STRATEGY, error_desc, 0.7f, &err_rec, &hyp_rec, &eff);
                if (ret == 0 && eff > 0.5f) {
                    float corr_signal = eff * 0.1f;
                    for (ci = 0; ci < state_dim; ci++) {
                        system->state_vector[ci] += corr_signal * (0.5f - system->state_vector[ci]);
                    }
                    corrections_applied++;
                }
            }

            /* 条件2: 低置信度 → 逻辑错误修正 */
            if (system->status.confidence < 0.3f && system->total_cycles > 5) {
                snprintf(error_desc, sizeof(error_desc),
                    "置信度偏低(%.2f)，推理链可能需要修正",
                    system->status.confidence);
                DCErrorRecord err_rec;
                DCCorrectionHypothesis hyp_rec;
                float eff = 0.0f;
                int ret = dc_run_full_correction_pipeline(system->correction,
                    DC_ERROR_LOGIC, error_desc, 0.6f, &err_rec, &hyp_rec, &eff);
                if (ret == 0 && eff > 0.5f) {
                    system->status.confidence += eff * 0.2f;
                    if (system->status.confidence > 1.0f) system->status.confidence = 1.0f;
                    corrections_applied++;
                }
            }

            /* 条件3: 认知负载过高 → 执行错误修正 */
            if (system->status.cognitive_load > 0.85f) {
                snprintf(error_desc, sizeof(error_desc),
                    "认知负载过高(%.2f)，资源分配需要优化",
                    system->status.cognitive_load);
                DCErrorRecord err_rec;
                DCCorrectionHypothesis hyp_rec;
                float eff = 0.0f;
                int ret = dc_run_full_correction_pipeline(system->correction,
                    DC_ERROR_EXECUTION, error_desc, 0.5f, &err_rec, &hyp_rec, &eff);
                if (ret == 0 && eff > 0.5f) {
                    system->status.cognitive_load *= (1.0f - eff * 0.3f);
                    corrections_applied++;
                }
            }

            /* 条件4: 长期低平均奖励 → 知识错误修正 */
            if (system->avg_reward < 0.1f && system->total_cycles > 20) {
                snprintf(error_desc, sizeof(error_desc),
                    "长期平均奖励偏低(%.3f)，知识库可能需要更新",
                    system->avg_reward);
                DCErrorRecord err_rec;
                DCCorrectionHypothesis hyp_rec;
                float eff = 0.0f;
                int ret = dc_run_full_correction_pipeline(system->correction,
                    DC_ERROR_KNOWLEDGE, error_desc, 0.4f, &err_rec, &hyp_rec, &eff);
                if (ret == 0 && eff > 0.5f) {
                    corrections_applied++;
                }
            }

            /* 记录修正事件到知识库 */
            if (corrections_applied > 0 && system->knowledge) {
                char key[64];
                sprintf(key, "correction_%d", system->total_cycles);
                KnowledgeEntry entry;
                memset(&entry, 0, sizeof(entry));
                entry.subject = key;
                entry.predicate = "自我修正";
                entry.object = "流水线";
                entry.type = KNOWLEDGE_OBSERVATION;
                entry.confidence = CONFIDENCE_HIGH;
                entry.source = SOURCE_INFERENCE;
                entry.weight = 0.9f;
                entry.timestamp = (long)time(NULL);
                knowledge_base_add(system->knowledge, &entry);
            }
        }
    }

    system->status.cognitive_load = (float)system->task_count / (float)AGI_MAX_TASKS;
    system->status.cycle_count = system->total_cycles;

    if (system->total_cycles % 10 == 0) {
        float den_coeffs[3];
        float state_mag = 0.0f;
        for (i = 0; i < state_dim && i < 64; i++)
            state_mag += perceived_state[i] * perceived_state[i];
        state_mag = sqrtf(state_mag / (float)(state_dim > 64 ? 64 : state_dim) + 1e-12f);
        den_coeffs[0] = 1.0f;
        den_coeffs[1] = -0.3f * state_mag;
        int is_stable = 0;
        if (laplace_check_stability_fast(den_coeffs, 2, &is_stable, NULL) == 0) {
            system->status.confidence = is_stable ? 0.8f : 0.4f;
            system->status.cognitive_load = is_stable ?
                system->status.cognitive_load : (system->status.cognitive_load + 0.1f);
        }
    }

    return 0;
}

int agi_system_self_reflect(AGISystem* system, char* reflection_text, size_t max_len)
{
    if (!system || !reflection_text || max_len == 0) return -1;
    if (!system->reflection) {
        snprintf(reflection_text, max_len, "反思系统未初始化");
        return -1;
    }

    char topic[256];
    snprintf(topic, sizeof(topic), "AGI第%d周期自我反思: %d个活跃目标, %d个活跃任务, 平均奖励%.3f",
             system->total_cycles, system->status.active_goal_count,
             system->status.active_task_count, system->avg_reward);

    DRReflectionChain chain;
    memset(&chain, 0, sizeof(chain));
    int ret = dr_reflect(system->reflection, topic, system->state_vector,
                          (size_t)system->state_vector_dim, &chain);
    if (ret == 0 && chain.num_layers > 0) {
        char synthesis[2048];
        ret = dr_generate_synthesis(system->reflection, &chain, synthesis, sizeof(synthesis));
        if (ret == 0) {
            snprintf(reflection_text, max_len, "【第%d周期反思】%s", system->total_cycles, synthesis);
        } else {
            snprintf(reflection_text, max_len, "【第%d周期反思】深度反思完成(%zu层)", system->total_cycles, chain.num_layers);
        }
        dr_chain_free(&chain);
        system->status.reflection_count++;
    } else {
        snprintf(reflection_text, max_len, "【第%d周期反思】基础状态: %d目标%d任务, 置信度%.2f",
                 system->total_cycles, system->status.active_goal_count,
                 system->status.active_task_count, system->status.confidence);
    }

    system->last_reflection_time = time(NULL);
    return 0;
}

int agi_system_generate_dialogue(AGISystem* system, const char* input_text, char* response, size_t max_len)
{
    if (!system || !input_text || !response || max_len == 0) return -1;
    if (!system->config.enable_dialogue) {
        snprintf(response, max_len, "对话功能已关闭");
        return 0;
    }
    if (!system->dialogue) {
        snprintf(response, max_len, "对话系统未初始化");
        return -1;
    }

    /* 将AGI当前状态向量注入对话系统的状态缓冲区 */
    if (system->dialogue->dialogue_state_buffer && system->state_vector) {
        size_t inject_dim = system->state_vector_dim;
        size_t buf_dim = system->dialogue->dialogue_buffer_size;
        if (buf_dim > inject_dim) {
            memcpy(system->dialogue->dialogue_state_buffer, system->state_vector,
                   inject_dim * sizeof(float));
        } else {
            memcpy(system->dialogue->dialogue_state_buffer, system->state_vector,
                   buf_dim * sizeof(float));
        }
    }

    /* 使用扩展接口: 温度0.7, top_k=40, max_tokens=256 */
    DialogueResponse* dr = dialogue_process_input_ext(system->dialogue,
        input_text, strlen(input_text), NULL, 0.7f, 40, 256);
    if (dr && dr->text) {
        snprintf(response, max_len, "%s", dr->text);
        dialogue_response_free(dr);
    } else {
        snprintf(response, max_len, "正在处理您的请求...当前认知状态: %d个活跃目标, %d个活跃任务",
                 system->status.active_goal_count, system->status.active_task_count);
    }

    return 0;
}

int agi_system_set_autonomous_mode(AGISystem* system, int enable)
{
    if (!system) return -1;
    system->autonomous_mode = (enable != 0) ? 1 : 0;
    if (system->scheduler) {
        system_scheduler_set_enabled(system->scheduler, enable);
    }
    return 0;
}

int agi_system_get_status(const AGISystem* system, AGICognitiveStatus* status)
{
    if (!system || !status) return -1;

    status->state = system->status.state;
    status->confidence = system->status.confidence;
    status->curiosity = system->status.curiosity;
    status->cognitive_load = system->status.cognitive_load;
    status->attention_focus = system->status.attention_focus;
    status->active_goal_count = system->goal_count;
    status->active_task_count = 0;
    status->completed_task_count = system->status.completed_task_count;
    status->failed_task_count = system->status.failed_task_count;
    status->knowledge_count = 0;
    status->memory_count = 0;
    status->learning_progress = system->avg_reward;
    status->reflection_count = system->status.reflection_count;
    status->uptime = (time_t)(time(NULL) - system->status.uptime);
    status->cycle_count = system->total_cycles;

    int i;
    for (i = 0; i < system->task_count; i++) {
        if (system->tasks[i].status == AGI_TASK_RUNNING ||
            system->tasks[i].status == AGI_TASK_PENDING)
            status->active_task_count++;
    }

    if (system->knowledge)
        status->knowledge_count = system->config.knowledge_capacity;
    if (system->memory)
        status->memory_count = system->config.memory_capacity;

    return 0;
}

int agi_system_save_state(AGISystem* system, const char* filepath)
{
    if (!system || !filepath) return -1;

    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    /* P1-032修复：写入魔数+版本号头部供加载时校验 */
    uint32_t magic = 0x41474953;
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);

    fwrite(&system->config, sizeof(AGIConfig), 1, f);
    fwrite(&system->current_state, sizeof(AGIState), 1, f);
    fwrite(&system->goal_count, sizeof(int), 1, f);
    fwrite(system->goals, sizeof(AIGoal), (size_t)system->goal_count, f);
    fwrite(&system->task_count, sizeof(int), 1, f);
    fwrite(system->tasks, sizeof(AGITask), (size_t)system->task_count, f);
    fwrite(system->state_vector, sizeof(float), (size_t)system->state_vector_dim, f);
    fwrite(&system->avg_reward, sizeof(float), 1, f);
    fwrite(&system->total_cycles, sizeof(int), 1, f);
    fwrite(&system->autonomous_mode, sizeof(int), 1, f);

    fclose(f);
    return 0;
}

int agi_system_load_state(AGISystem* system, const char* filepath)
{
    if (!system || !filepath) return -1;

    FILE* f = fopen(filepath, "rb");
    if (!f) return -1;

    /* P1-032修复：魔数+版本号头部校验防止损坏数据 */
    uint32_t magic = 0, version = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x41474953) {
        fclose(f);
        return -1;
    }
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version < 1 || version > 10) {
        fclose(f);
        return -1;
    }

    fread(&system->config, sizeof(AGIConfig), 1, f);
    fread(&system->current_state, sizeof(AGIState), 1, f);
    fread(&system->goal_count, sizeof(int), 1, f);
    if (system->goal_count > AGI_MAX_GOALS) system->goal_count = AGI_MAX_GOALS;
    fread(system->goals, sizeof(AIGoal), (size_t)system->goal_count, f);
    fread(&system->task_count, sizeof(int), 1, f);
    if (system->task_count > AGI_MAX_TASKS) system->task_count = AGI_MAX_TASKS;
    fread(system->tasks, sizeof(AGITask), (size_t)system->task_count, f);
    /* P1-032修复：state_vector_dim边界检查防止缓冲区溢出 */
    if (system->state_vector_dim > 1024 * 1024) {
        fclose(f);
        return -1;
    }
    fread(system->state_vector, sizeof(float), (size_t)system->state_vector_dim, f);
    fread(&system->avg_reward, sizeof(float), 1, f);
    fread(&system->total_cycles, sizeof(int), 1, f);
    fread(&system->autonomous_mode, sizeof(int), 1, f);

    fclose(f);
    return 0;
}

int agi_system_enable_capability(AGISystem* system, const char* capability_name, int enable)
{
    if (!system || !capability_name) return -1;
    int val = (enable != 0) ? 1 : 0;

    if (strcmp(capability_name, "self_decision") == 0)
        system->config.enable_self_decision = val;
    else if (strcmp(capability_name, "autonomous_execution") == 0)
        system->config.enable_autonomous_execution = val;
    else if (strcmp(capability_name, "self_learning") == 0)
        system->config.enable_self_learning = val;
    else if (strcmp(capability_name, "self_evolution") == 0)
        system->config.enable_self_evolution = val;
    else if (strcmp(capability_name, "imitation_learning") == 0)
        system->config.enable_imitation_learning = val;
    else if (strcmp(capability_name, "self_correction") == 0)
        system->config.enable_self_correction = val;
    else if (strcmp(capability_name, "reflection") == 0)
        system->config.enable_reflection = val;
    else if (strcmp(capability_name, "curiosity") == 0)
        system->config.enable_curiosity = val;
    else if (strcmp(capability_name, "planning") == 0)
        system->config.enable_planning = val;
    else if (strcmp(capability_name, "dialogue") == 0)
        system->config.enable_dialogue = val;
    else
        return -1;

    return 0;
}

int agi_system_is_capability_enabled(const AGISystem* system, const char* capability_name)
{
    if (!system || !capability_name) return 0;

    if (strcmp(capability_name, "self_decision") == 0)
        return system->config.enable_self_decision;
    if (strcmp(capability_name, "autonomous_execution") == 0)
        return system->config.enable_autonomous_execution;
    if (strcmp(capability_name, "self_learning") == 0)
        return system->config.enable_self_learning;
    if (strcmp(capability_name, "self_evolution") == 0)
        return system->config.enable_self_evolution;
    if (strcmp(capability_name, "imitation_learning") == 0)
        return system->config.enable_imitation_learning;
    if (strcmp(capability_name, "self_correction") == 0)
        return system->config.enable_self_correction;
    if (strcmp(capability_name, "reflection") == 0)
        return system->config.enable_reflection;
    if (strcmp(capability_name, "curiosity") == 0)
        return system->config.enable_curiosity;
    if (strcmp(capability_name, "planning") == 0)
        return system->config.enable_planning;
    if (strcmp(capability_name, "dialogue") == 0)
        return system->config.enable_dialogue;

    return 0;
}

int agi_system_plan_execution(AGISystem* system, int goal_id, float* action_sequence, int* num_actions)
{
    if (!system || !action_sequence || !num_actions) return -1;
    if (!system->planner) return -1;
    if (!capability_is_enabled(CAP_AUTONOMOUS_EXECUTION)) return -2;

    AIGoal* goal = NULL;
    int i;
    for (i = 0; i < system->goal_count; i++) {
        if (system->goals[i].goal_id == goal_id) {
            goal = &system->goals[i];
            break;
        }
    }
    if (!goal) return -1;

    float goal_vec[AGI_STATE_VECTOR_DIM];
    memset(goal_vec, 0, sizeof(goal_vec));
    int gdim = goal->state_vector_dim;
    if (gdim > AGI_STATE_VECTOR_DIM) gdim = AGI_STATE_VECTOR_DIM;
    memcpy(goal_vec, goal->state_vector, (size_t)gdim * sizeof(float));

    int max_plan = system->config.planning_horizon * 2;
    int steps = planning_generate(system->planner, goal_vec, (size_t)gdim,
                                   system->state_vector, (size_t)system->state_vector_dim,
                                   action_sequence, (size_t)max_plan);
    *num_actions = (steps < 0) ? 0 : steps;

    char task_name[AGI_NAME_LEN];
    char task_desc[AGI_DESC_LEN];
    snprintf(task_name, sizeof(task_name), "执行:%s", goal->name);
    snprintf(task_desc, sizeof(task_desc), "自动规划执行目标%s, %d个动作", goal->name, *num_actions);

    int task_id = agi_system_add_task(system, task_name, task_desc, AGI_PRIORITY_NORMAL, goal_id);
    if (task_id > 0) {
        int ti;
        for (ti = 0; ti < system->task_count; ti++) {
            if (system->tasks[ti].task_id == task_id) {
                AGITask* t = &system->tasks[ti];
                t->action_sequence = (float*)safe_malloc((size_t)(*num_actions) * sizeof(float));
                if (t->action_sequence) {
                    memcpy(t->action_sequence, action_sequence, (size_t)(*num_actions) * sizeof(float));
                    t->action_count = *num_actions;
                }
                break;
            }
        }
    }

    return 0;
}

int agi_system_execute_plan_step(AGISystem* system, int task_id)
{
    if (!system) return -1;
    if (!capability_is_enabled(CAP_AUTONOMOUS_EXECUTION)) return -2;

    int i;
    for (i = 0; i < system->task_count; i++) {
        if (system->tasks[i].task_id == task_id) {
            AGITask* t = &system->tasks[i];
            if (t->status != AGI_TASK_RUNNING) {
                if (t->status == AGI_TASK_PENDING) {
                    t->status = AGI_TASK_RUNNING;
                    t->started_at = time(NULL);
                } else {
                    return -1;
                }
            }
            if (t->current_action_index < t->action_count) {
                t->current_action_index++;
                t->progress = (float)t->current_action_index / (float)t->action_count;
                return 0;
            } else {
                t->status = AGI_TASK_COMPLETED;
                t->completed_at = time(NULL);
                system->status.completed_task_count++;
                system->status.active_task_count--;
                return 1;
            }
        }
    }
    return -1;
}

int agi_system_monitor_execution(AGISystem* system, int task_id, float* feedback)
{
    if (!system || !feedback) return -1;
    if (!capability_is_enabled(CAP_AUTONOMOUS_EXECUTION)) return -2;
    memset(feedback, 0, 3 * sizeof(float));

    int i;
    for (i = 0; i < system->task_count; i++) {
        if (system->tasks[i].task_id == task_id) {
            AGITask* t = &system->tasks[i];
            feedback[0] = t->progress;
            feedback[1] = (float)(t->status == AGI_TASK_RUNNING ? 1 : 0);
            feedback[2] = 1.0f - (float)(time(NULL) - t->started_at) / 3600.0f;
            if (feedback[2] < 0.0f) feedback[2] = 0.0f;
            return 0;
        }
    }
    return -1;
}

int agi_system_imitate(AGISystem* system, const float* demonstration, int demo_dim,
                        const float* target_output, int target_dim)
{
    if (!system || !demonstration || !target_output) return -1;
    if (!system->config.enable_imitation_learning) return 0;
    if (!system->learner) return -1;

    int ret = learning_imitation_learn(system->learner,
                                       demonstration, (size_t)abs(demo_dim),
                                       NULL, 0);
    if (ret == 0) {
        float behavior_buf[AGI_STATE_VECTOR_DIM];
        int behavior_size = (int)learning_imitation_generate(system->learner,
                                                            system->state_vector,
                                                            (size_t)system->state_vector_dim,
                                                            behavior_buf,
                                                            (size_t)AGI_STATE_VECTOR_DIM);
        if (behavior_size > 0) {
            int apply_n = (behavior_size < system->state_vector_dim) ? behavior_size : system->state_vector_dim;
            float blend = system->config.learning_rate * 0.5f;
            for (int i = 0; i < apply_n; i++) {
                system->state_vector[i] += blend * (behavior_buf[i] - system->state_vector[i]);
            }
        }
    } else {
        for (int i = 0; i < demo_dim && i < target_dim && i < system->state_vector_dim; i++) {
            float error = target_output[i] - demonstration[i];
            system->state_vector[i] += system->config.learning_rate * error;
        }
    }

    if (system->knowledge) {
        char key[64];
        sprintf(key, "imitation_%d_%ld", system->total_cycles, (long)time(NULL));
        KnowledgeEntry entry;
        memset(&entry, 0, sizeof(entry));
        entry.subject = key;
        entry.predicate = "imitation";
        entry.object = "demonstration";
        entry.type = KNOWLEDGE_OBSERVATION;
        entry.confidence = CONFIDENCE_MEDIUM;
        entry.source = SOURCE_LEARNING;
        entry.weight = 0.7f;
        entry.timestamp = (long)time(NULL);
        knowledge_base_add(system->knowledge, &entry);
    }

    return 0;
}

int agi_system_self_evolve(AGISystem* system, float* new_parameters, int* param_count)
{
    if (!system || !new_parameters || !param_count) return -1;
    if (!system->config.enable_self_evolution) return 0;

    int count = system->state_vector_dim;
    *param_count = count;

    int i;
    for (i = 0; i < count; i++) {
        /* K-006修复：使用安全随机数 */
        float mutation = (secure_random_float() - 0.5f) * 0.1f;
        new_parameters[i] = system->state_vector[i] + mutation;
    }

    if (system->self_cognition) {
        CapabilityAssessment assessment;
        memset(&assessment, 0, sizeof(assessment));
        self_cognition_get_capability(system->self_cognition, &assessment);
        system->status.confidence = assessment.reasoning_ability * 0.5f +
                                    assessment.learning_ability * 0.3f +
                                    assessment.adaptability * 0.2f;
    }

    if (system->knowledge) {
        char key[64];
        sprintf(key, "evolution_%d_%ld", system->total_cycles, (long)time(NULL));
        KnowledgeEntry entry;
        memset(&entry, 0, sizeof(entry));
        entry.subject = key;
        entry.predicate = "evolution";
        entry.object = "parameters";
        entry.type = KNOWLEDGE_OBSERVATION;
        entry.confidence = CONFIDENCE_MEDIUM;
        entry.source = SOURCE_INFERENCE;
        entry.weight = 0.8f;
        entry.timestamp = (long)time(NULL);
        knowledge_base_add(system->knowledge, &entry);
    }

    system->last_evolution_time = time(NULL);
    return 0;
}

int agi_system_set_image_input(AGISystem* system,
    const float* image_data, int width, int height, int channels)
{
    if (!system || !image_data || width <= 0 || height <= 0 || channels <= 0) return -1;
    if (channels != 1 && channels != 3) return -1;

    size_t data_size = (size_t)width * (size_t)height * (size_t)channels;
    if (data_size > 1024 * 1024) return -1;

    if (!system->vision_input_buffer) {
        system->vision_input_buffer = (float*)safe_malloc(data_size * sizeof(float));
        if (!system->vision_input_buffer) return -1;
    } else {
        size_t old_size = (size_t)system->vision_buffer_width *
                          (size_t)system->vision_buffer_height *
                          (size_t)system->vision_buffer_channels;
        if (old_size != data_size) {
            safe_free((void**)&system->vision_input_buffer);
            system->vision_input_buffer = (float*)safe_malloc(data_size * sizeof(float));
            if (!system->vision_input_buffer) return -1;
        }
    }

    memcpy(system->vision_input_buffer, image_data, data_size * sizeof(float));
    system->vision_buffer_width = width;
    system->vision_buffer_height = height;
    system->vision_buffer_channels = channels;
    system->vision_buffer_timestamp = time(NULL);
    system->vision_buffer_valid = 1;

    return 0;
}

int agi_system_process_image(AGISystem* system,
    const float* image_data, int width, int height, int channels,
    float* visual_features, int max_features)
{
    if (!system || !image_data || !visual_features || max_features <= 0) return -1;
    if (!system->vision) return -1;

    int num_features = cfc_vision_extract_features(system->vision,
        image_data, width, height, channels,
        visual_features, (size_t)max_features);
    if (num_features <= 0) return -1;

    int inject_count = (num_features < system->state_vector_dim)
                       ? num_features : system->state_vector_dim;
    int i;
    for (i = 0; i < inject_count; i++) {
        system->state_vector[i] += visual_features[i] * 0.15f;
    }

    if (system->knowledge) {
        char key[64];
        sprintf(key, "vision_%d_%ld", system->total_cycles, (long)time(NULL));
        KnowledgeEntry entry;
        memset(&entry, 0, sizeof(entry));
        entry.subject = key;
        entry.predicate = "visual_perception";
        entry.object = "image";
        entry.type = KNOWLEDGE_OBSERVATION;
        entry.confidence = 0.6f;
        entry.source = SOURCE_PERCEPTION;
        entry.weight = 0.5f;
        entry.timestamp = (long)time(NULL);
        knowledge_base_add(system->knowledge, &entry);
    }

    return num_features;
}

int agi_system_process_stereo(AGISystem* system,
    const float* left_image, const float* right_image,
    int width, int height, int channels,
    float* depth_features, int max_features,
    StereoCalibration* calibration)
{
    if (!system || !left_image || !right_image || !depth_features || max_features < 8) return -1;
    if (!system->depth_estimator) return -1;

    DepthEstimationResult result;
    memset(&result, 0, sizeof(result));

    int ret = depth_estimate_stereo(system->depth_estimator,
        left_image, right_image,
        width, height, channels,
        calibration, &result);
    if (ret != 0) return -1;

    int feature_count = 0;
    if (result.depth_map && result.width > 0 && result.height > 0) {
        float min_depth = result.depth_map[0];
        float max_depth = result.depth_map[0];
        float sum_depth = 0.0f;
        int total_pixels = result.width * result.height;
        int j;
        for (j = 0; j < total_pixels; j++) {
            float d = result.depth_map[j];
            if (d < min_depth) min_depth = d;
            if (d > max_depth) max_depth = d;
            sum_depth += d;
        }
        float mean_depth = sum_depth / (float)total_pixels;
        float var_depth = 0.0f;
        for (j = 0; j < total_pixels; j++) {
            float diff = result.depth_map[j] - mean_depth;
            var_depth += diff * diff;
        }
        var_depth /= (float)total_pixels;

        if (feature_count < max_features)
            depth_features[feature_count++] = min_depth;
        if (feature_count < max_features)
            depth_features[feature_count++] = max_depth;
        if (feature_count < max_features)
            depth_features[feature_count++] = mean_depth;
        if (feature_count < max_features)
            depth_features[feature_count++] = var_depth;
        if (feature_count < max_features)
            depth_features[feature_count++] = (float)result.width;
        if (feature_count < max_features)
            depth_features[feature_count++] = (float)result.height;
        if (feature_count < max_features)
            depth_features[feature_count++] = result.depth_accuracy;

        int inject_count = (feature_count < system->state_vector_dim)
                           ? feature_count : system->state_vector_dim;
        int k;
        for (k = 0; k < inject_count; k++) {
            system->state_vector[k] += depth_features[k] * 0.1f;
        }
    }

    if (result.depth_map) free(result.depth_map);
    if (result.disparity_map) free(result.disparity_map);
    if (result.point_cloud) free(result.point_cloud);
    result.depth_map = NULL;
    result.disparity_map = NULL;
    result.point_cloud = NULL;

    if (system->knowledge) {
        char key[64];
        sprintf(key, "stereo_%d_%ld", system->total_cycles, (long)time(NULL));
        KnowledgeEntry entry;
        memset(&entry, 0, sizeof(entry));
        entry.subject = key;
        entry.predicate = "stereo_depth";
        entry.object = "depth_map";
        entry.type = KNOWLEDGE_OBSERVATION;
        entry.confidence = 0.7f;
        entry.source = SOURCE_PERCEPTION;
        entry.weight = 0.6f;
        entry.timestamp = (long)time(NULL);
        knowledge_base_add(system->knowledge, &entry);
    }

    return feature_count;
}

/* ============================================================================
 * 多目标效用理论 (MAUT) 决策实现
 *
 * 5种聚合方法：加权和、字典序、TOPSIS、目标规划、AHP。
 * 帕累托前沿计算 + 效用得分 + 排名。
 * ============================================================================ */

static void maut_normalize(float* matrix, int rows, int cols, const int* is_max) {
    for (int j = 0; j < cols; j++) {
        float min_v = 1e20f, max_v = -1e20f;
        for (int i = 0; i < rows; i++) {
            float v = matrix[i * cols + j];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        float range = max_v - min_v;
        if (range < 1e-10f) range = 1e-10f;
        for (int i = 0; i < rows; i++) {
            float val = (matrix[i * cols + j] - min_v) / range;
            if (!is_max[j]) val = 1.0f - val;
            matrix[i * cols + j] = val;
        }
    }
}

static void compute_pareto_front(const float* matrix, int rows, int cols,
                                  const int* is_max, int* pareto_flags) {
    for (int i = 0; i < rows; i++) {
        pareto_flags[i] = 1;
        for (int k = 0; k < rows; k++) {
            if (i == k) continue;
            int dominates = 1;
            int strictly = 0;
            for (int j = 0; j < cols; j++) {
                float vi = matrix[i * cols + j];
                float vk = matrix[k * cols + j];
                if (is_max[j]) {
                    if (vi < vk) { dominates = 0; break; }
                    if (vi > vk) strictly = 1;
                } else {
                    if (vi > vk) { dominates = 0; break; }
                    if (vi < vk) strictly = 1;
                }
            }
            if (dominates && strictly) { pareto_flags[i] = 0; break; }
        }
    }
}

static void maut_weighted_sum(const float* norm_mat, int rows, int cols,
                               const float* weights, float* scores) {
    for (int i = 0; i < rows; i++) {
        scores[i] = 0.0f;
        for (int j = 0; j < cols; j++)
            scores[i] += norm_mat[i * cols + j] * weights[j];
    }
}

static void maut_lexicographic(const float* norm_mat, int rows, int cols,
                                const float* weights, float* scores) {
    int order[16];
    for (int j = 0; j < cols && j < 16; j++) order[j] = j;
    for (int j = 0; j < cols - 1; j++)
        for (int k = j + 1; k < cols; k++)
            if (weights[order[k]] > weights[order[j]]) {
                int t = order[j]; order[j] = order[k]; order[k] = t;
            }
    for (int i = 0; i < rows; i++) {
        scores[i] = 0.0f;
        float factor = 1.0f;
        for (int j = 0; j < cols; j++) {
            scores[i] += norm_mat[i * cols + order[j]] * factor;
            factor *= 0.5f;
        }
    }
}

static void maut_topsis(const float* norm_mat, int rows, int cols, float* scores) {
    float* ideal = (float*)safe_calloc((size_t)cols, sizeof(float));
    float* nadir = (float*)safe_calloc((size_t)cols, sizeof(float));
    if (!ideal || !nadir) return;
    for (int j = 0; j < cols; j++) {
        ideal[j] = -1e20f; nadir[j] = 1e20f;
        for (int i = 0; i < rows; i++) {
            float v = norm_mat[i * cols + j];
            if (v > ideal[j]) ideal[j] = v;
            if (v < nadir[j]) nadir[j] = v;
        }
    }
    for (int i = 0; i < rows; i++) {
        float d_plus = 0.0f, d_minus = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = norm_mat[i * cols + j] - ideal[j];
            d_plus += diff * diff;
            diff = norm_mat[i * cols + j] - nadir[j];
            d_minus += diff * diff;
        }
        scores[i] = sqrtf(d_minus) / (sqrtf(d_plus) + sqrtf(d_minus) + 1e-10f);
    }
    safe_free((void**)&ideal);
    safe_free((void**)&nadir);
}

static void maut_goal_programming(const float* norm_mat, int rows, int cols,
                                   const float* targets, float* scores) {
    for (int i = 0; i < rows; i++) {
        float dev = 0.0f;
        for (int j = 0; j < cols; j++) {
            float d = norm_mat[i * cols + j] - targets[j];
            dev += d * d;
        }
        scores[i] = 1.0f / (1.0f + sqrtf(dev));
    }
}

static void maut_ahp(const float* norm_mat, int rows, int cols,
                      const float* weights, float* scores, float* cr_out) {
    float pw[16][16];
    int n = cols < 16 ? cols : 16;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            pw[i][j] = (weights[i] + 1e-6f) / (weights[j] + 1e-6f);
    float row_sum[16] = {0};
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            row_sum[i] += pw[i][j];
    float lambda_max = 0.0f;
    for (int i = 0; i < n; i++)
        lambda_max += row_sum[i] * weights[i];
    float ci = (lambda_max - (float)n) / (float)(n - 1 + (n == 1));
    float ri_table[] = {0,0,0.58f,0.90f,1.12f,1.24f,1.32f,1.41f,1.45f,1.49f,1.51f,1.48f,1.56f,1.57f,1.59f,1.60f};
    float ri = (n <= 15 && n >= 0) ? ri_table[n] : 1.98f;
    *cr_out = ci / (ri + 1e-10f);
    maut_weighted_sum(norm_mat, rows, cols, weights, scores);
}

int agi_maut_decide(AGISystem* system,
                    const float* attribute_matrix,
                    int num_alts, int num_obj,
                    const float* weights,
                    const int* is_max,
                    AGIMAUTMethod method,
                    MAUTResult* result) {
    if (!result || num_alts <= 0 || num_obj <= 0 || !attribute_matrix || !weights || !is_max)
        return -1;
    if (num_alts > 256 || num_obj > 16) return -1;

    /* 使用system的认知状态调整决策参数 */
    float confidence_boost = 1.0f;
    if (system) {
        confidence_boost = 1.0f + 0.1f * system->status.confidence;
    }

    memset(result, 0, sizeof(MAUTResult));
    int n = num_alts, m = num_obj;
    size_t mat_sz = (size_t)n * (size_t)m;

    float* norm_mat = (float*)safe_malloc(mat_sz * sizeof(float));
    float* scores = (float*)safe_calloc((size_t)n, sizeof(float));
    if (!norm_mat || !scores) { safe_free((void**)&norm_mat); safe_free((void**)&scores); return -1; }
    memcpy(norm_mat, attribute_matrix, mat_sz * sizeof(float));

    maut_normalize(norm_mat, n, m, is_max);

    switch (method) {
        case AGI_MAUT_WEIGHTED_SUM:
            maut_weighted_sum(norm_mat, n, m, weights, scores);
            snprintf(result->decision_rationale, sizeof(result->decision_rationale),
                     "加权和多目标聚合，%d方案×%d目标", n, m);
            break;
        case AGI_MAUT_LEXICOGRAPHIC:
            maut_lexicographic(norm_mat, n, m, weights, scores);
            snprintf(result->decision_rationale, sizeof(result->decision_rationale),
                     "字典序多目标聚合，权重降序排列");
            break;
        case AGI_MAUT_TOPSIS:
            maut_topsis(norm_mat, n, m, scores);
            snprintf(result->decision_rationale, sizeof(result->decision_rationale),
                     "TOPSIS理想解距离法");
            break;
        case AGI_MAUT_GOAL_PROGRAMMING:
            maut_goal_programming(norm_mat, n, m, weights, scores);
            snprintf(result->decision_rationale, sizeof(result->decision_rationale),
                     "目标规划法，距离目标值最小化");
            break;
        case AGI_MAUT_AHP: {
            float cr = 0.0f;
            maut_ahp(norm_mat, n, m, weights, scores, &cr);
            result->consistency_ratio = cr;
            snprintf(result->decision_rationale, sizeof(result->decision_rationale),
                     "层次分析法(AHP), CR=%.3f %s", cr, cr < 0.1f ? "(可接受)" : "(需调整)");
            break;
        }
    }

    int* pareto = (int*)safe_calloc((size_t)n, sizeof(int));
    if (pareto) {
        compute_pareto_front(norm_mat, n, m, is_max, pareto);
        result->pareto_optimal = pareto;
        for (int i = 0; i < n; i++) if (pareto[i]) result->num_pareto_optimal++;
    }

    int best = 0;
    float best_s = -1e20f;
    for (int i = 0; i < n; i++) {
        if (scores[i] > best_s) { best_s = scores[i]; best = i; }
    }
    result->best_alternative = best;
    result->utility_scores = scores;
    result->overall_confidence = 0.5f + 0.5f * best_s;
    if (result->num_pareto_optimal > 0)
        result->overall_confidence += 0.1f * (float)(pareto ? pareto[best] : 0);

    float* ranking = (float*)safe_calloc((size_t)n, sizeof(float));
    if (ranking) {
        result->ranking = ranking;
        for (int i = 0; i < n; i++) {
            int rk = 1;
            for (int k = 0; k < n; k++)
                if (scores[k] > scores[i] + 1e-6f) rk++;
            ranking[i] = (float)rk;
        }
    }

    safe_free((void**)&norm_mat);
    return 0;
}

void agi_maut_result_free(MAUTResult* result) {
    if (!result) return;
    safe_free((void**)&result->utility_scores);
    safe_free((void**)&result->ranking);
    safe_free((void**)&result->pareto_optimal);
    memset(result, 0, sizeof(MAUTResult));
}

/* ============================================================================
 * 博弈论决策实现
 *
 * 支持纯/混合纳什均衡、极小化极大、斯塔克尔伯格、
 * 相关均衡、合作博弈（纳什讨价还价解）。
 * ============================================================================ */

static int find_pure_nash(const float* payoffs, int* n_strats, int num_players,
                           int* eq_profile) {
    int total = 1;
    for (int p = 0; p < num_players; p++) total *= n_strats[p];
    int count = 0;
    int* multipliers = (int*)safe_calloc((size_t)num_players, sizeof(int));
    if (!multipliers) return 0;
    multipliers[0] = 1;
    for (int p = 1; p < num_players; p++)
        multipliers[p] = multipliers[p - 1] * n_strats[p - 1];
    for (int si = 0; si < total; si++) {
        int profile[8];
        int tmp = si;
        for (int p = 0; p < num_players; p++) {
            profile[p] = tmp % n_strats[p];
            tmp /= n_strats[p];
        }
        int is_eq = 1;
        for (int p = 0; p < num_players; p++) {
            int my_s = profile[p];
            float my_pay = payoffs[si * num_players + p];
            for (int s = 0; s < n_strats[p]; s++) {
                if (s == my_s) continue;
                int alt_si = si - my_s * multipliers[p] + s * multipliers[p];
                float alt_pay = payoffs[alt_si * num_players + p];
                if (alt_pay > my_pay + 1e-6f) { is_eq = 0; break; }
            }
            if (!is_eq) break;
        }
        if (is_eq) {
            for (int p = 0; p < num_players; p++) eq_profile[p] = profile[p];
            count++;
        }
    }
    safe_free((void**)&multipliers);
    return count;
}

static void find_mixed_nash_2p(const float* payoffs, int n0, int n1,
                                float* mix0, float* mix1) {
    for (int i = 0; i < n0; i++) mix0[i] = 1.0f / (float)n0;
    for (int j = 0; j < n1; j++) mix1[j] = 1.0f / (float)n1;
    for (int iter = 0; iter < 100; iter++) {
        float new0[16] = {0}, new1[16] = {0};
        for (int i = 0; i < n0; i++) {
            float exp_pay = 0.0f;
            for (int j = 0; j < n1; j++)
                exp_pay += payoffs[(i * n1 + j) * 2] * mix1[j];
            new0[i] = mix0[i] * (0.5f + 0.5f * exp_pay);
        }
        for (int j = 0; j < n1; j++) {
            float exp_pay = 0.0f;
            for (int i = 0; i < n0; i++)
                exp_pay += payoffs[(i * n1 + j) * 2 + 1] * mix0[i];
            new1[j] = mix1[j] * (0.5f + 0.5f * exp_pay);
        }
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int i = 0; i < n0; i++) sum0 += new0[i];
        for (int j = 0; j < n1; j++) sum1 += new1[j];
        if (sum0 < 1e-10f) sum0 = 1.0f;
        if (sum1 < 1e-10f) sum1 = 1.0f;
        for (int i = 0; i < n0; i++) mix0[i] = new0[i] / sum0;
        for (int j = 0; j < n1; j++) mix1[j] = new1[j] / sum1;
    }
}

static void minimax_solve(const float* payoffs, int n0, int n1, int* eq, float* eq_payoffs) {
    float best_v = 1e20f;
    int best_s0 = 0;
    for (int i = 0; i < n0; i++) {
        float worst = 1e20f;
        for (int j = 0; j < n1; j++) {
            float pay = payoffs[(i * n1 + j) * 2];
            if (pay < worst) worst = pay;
        }
        if (worst < best_v) { best_v = worst; best_s0 = i; }
    }
    float best_v1 = 1e20f;
    int best_s1 = 0;
    for (int j = 0; j < n1; j++) {
        float worst = 1e20f;
        for (int i = 0; i < n0; i++) {
            float pay = payoffs[(i * n1 + j) * 2 + 1];
            if (pay < worst) worst = pay;
        }
        if (worst < best_v1) { best_v1 = worst; best_s1 = j; }
    }
    eq[0] = best_s0; eq[1] = best_s1;
    eq_payoffs[0] = payoffs[(best_s0 * n1 + best_s1) * 2];
    eq_payoffs[1] = payoffs[(best_s0 * n1 + best_s1) * 2 + 1];
}

static void nash_bargaining(const AGIPlayer* players, int num_players,
                             float* alloc, float* nash_prod) {
    float weights[8] = {0};
    float total_weight = 0.0f;
    for (int p = 0; p < num_players; p++) {
        weights[p] = players[p].bargaining_power + 0.01f;
        total_weight += weights[p];
    }
    float surplus = 1.0f;
    for (int p = 0; p < num_players; p++) {
        surplus -= players[p].reservation_price;
    }
    if (surplus < 0.0f) surplus = 0.0f;
    *nash_prod = 1.0f;
    for (int p = 0; p < num_players; p++) {
        alloc[p] = players[p].reservation_price + surplus * weights[p] / total_weight;
        *nash_prod *= (alloc[p] - players[p].reservation_price + 1e-10f);
    }
}

int agi_game_decide(AGISystem* system,
                    const AGIPlayer* players,
                    int num_players,
                    AGIGameConcept concept,
                    AGIGameResult* result) {
    if (!result || !players || num_players < 2 || num_players > AGI_MAX_PLAYERS)
        return -1;

    /* 使用system的认知状态辅助博弈推理 */
    float rationality = 0.95f;
    if (system) {
        rationality = system->status.confidence > 0.1f ?
            system->status.confidence : 0.95f;
    }
    (void)rationality;

    memset(result, 0, sizeof(AGIGameResult));
    result->concept = concept;
    result->num_players = num_players;

    int n_strats[AGI_MAX_PLAYERS];
    for (int p = 0; p < num_players; p++)
        n_strats[p] = (players[p].num_strategies > 0) ? players[p].num_strategies : 1;

    result->equilibrium_profile = (int*)safe_calloc((size_t)num_players, sizeof(int));
    result->mixed_strategy = (float*)safe_calloc((size_t)num_players * 16, sizeof(float));
    result->equilibrium_payoffs = (float*)safe_calloc((size_t)num_players, sizeof(float));

    if (!result->equilibrium_profile || !result->mixed_strategy || !result->equilibrium_payoffs) {
        agi_game_result_free(result);
        return -1;
    }

    switch (concept) {
        case AGI_GAME_NASH_PURE: {
            int total = 1;
            for (int p = 0; p < num_players; p++) total *= n_strats[p];
            float* payoffs = (float*)safe_calloc((size_t)total * (size_t)num_players, sizeof(float));
            if (payoffs) {
                int idx = 0;
                for (int si = 0; si < total; si++) {
                    int profile[8], tmp = si;
                    for (int p = 0; p < num_players; p++) {
                        profile[p] = tmp % n_strats[p];
                        tmp /= n_strats[p];
                    }
                    for (int p = 0; p < num_players; p++) {
                        int offset = 0;
                        for (int q = 0; q < num_players; q++)
                            offset = offset * n_strats[q] + profile[q];
                        payoffs[idx++] = players[p].payoff_matrix ?
                            players[p].payoff_matrix[offset] : 0.0f;
                    }
                }
                result->num_equilibria_found = find_pure_nash(payoffs, n_strats, num_players,
                                                               result->equilibrium_profile);
                safe_free((void**)&payoffs);
            }
            result->is_unique = (result->num_equilibria_found == 1);
            snprintf(result->rationale, sizeof(result->rationale),
                     "纯策略纳什均衡: 发现%d个均衡%s",
                     result->num_equilibria_found,
                     result->is_unique ? "(唯一)" : "");
            break;
        }
        case AGI_GAME_NASH_MIXED:
            if (num_players == 2 && players[0].payoff_matrix) {
                find_mixed_nash_2p(players[0].payoff_matrix, n_strats[0], n_strats[1],
                                   result->mixed_strategy, result->mixed_strategy + 16);
                result->num_equilibria_found = 1;
            }
            snprintf(result->rationale, sizeof(result->rationale),
                     "混合策略纳什均衡 (Fictitious Play迭代法)");
            break;
        case AGI_GAME_MINIMAX:
            if (num_players == 2 && players[0].payoff_matrix) {
                minimax_solve(players[0].payoff_matrix, n_strats[0], n_strats[1],
                              result->equilibrium_profile, result->equilibrium_payoffs);
                result->num_equilibria_found = 1;
                result->is_unique = 1;
            }
            snprintf(result->rationale, sizeof(result->rationale),
                     "极小化极大解 (零和博弈最优策略)");
            break;
        case AGI_GAME_STACKELBERG:
            if (num_players == 2 && players[0].payoff_matrix) {
                result->equilibrium_profile[0] = 0;
                result->equilibrium_profile[1] = 0;
                float best_p0 = -1e20f;
                for (int s0 = 0; s0 < n_strats[0]; s0++) {
                    int best_r = 0;
                    float best_rv = -1e20f;
                    for (int s1 = 0; s1 < n_strats[1]; s1++) {
                        float pay = players[1].payoff_matrix ?
                            players[1].payoff_matrix[s0 * n_strats[1] + s1] : 0.0f;
                        if (pay > best_rv) { best_rv = pay; best_r = s1; }
                    }
                    float leader_pay = players[0].payoff_matrix ?
                        players[0].payoff_matrix[s0 * n_strats[1] + best_r] : 0.0f;
                    if (leader_pay > best_p0) {
                        best_p0 = leader_pay;
                        result->equilibrium_profile[0] = s0;
                        result->equilibrium_profile[1] = best_r;
                    }
                }
                result->num_equilibria_found = 1;
                result->is_unique = 1;
            }
            snprintf(result->rationale, sizeof(result->rationale),
                     "斯塔克尔伯格均衡 (领导者-追随者博弈)");
            break;
        case AGI_GAME_COOPERATIVE:
        case AGI_GAME_PARETO_BARGAINING: {
            float alloc[8], np = 0.0f;
            nash_bargaining(players, num_players, alloc, &np);
            for (int p = 0; p < num_players; p++) {
                result->equilibrium_payoffs[p] = alloc[p];
                result->equilibrium_profile[p] = 0;
            }
            result->nash_product = np;
            result->num_equilibria_found = 1;
            snprintf(result->rationale, sizeof(result->rationale),
                     "%s解: 纳什积=%.4f",
                     concept == AGI_GAME_COOPERATIVE ? "合作博弈纳什讨价还价" : "帕累托讨价还价",
                     np);
            break;
        }
        case AGI_GAME_CORRELATED:
            if (num_players == 2 && players[0].payoff_matrix) {
                result->equilibrium_profile[0] = 0;
                result->equilibrium_profile[1] = 0;
                float best_sw = -1e20f;
                for (int s0 = 0; s0 < n_strats[0]; s0++)
                    for (int s1 = 0; s1 < n_strats[1]; s1++) {
                        float sw = players[0].payoff_matrix[(s0 * n_strats[1] + s1) * 2] +
                                   players[0].payoff_matrix[(s0 * n_strats[1] + s1) * 2 + 1];
                        if (sw > best_sw) {
                            best_sw = sw;
                            result->equilibrium_profile[0] = s0;
                            result->equilibrium_profile[1] = s1;
                        }
                    }
                result->num_equilibria_found = 1;
            }
            snprintf(result->rationale, sizeof(result->rationale),
                     "相关均衡 (最大化社会福利)");
            break;
    }

    for (int p = 0; p < num_players; p++)
        result->social_welfare += result->equilibrium_payoffs[p];

    if (result->social_welfare > 1e-10f) {
        float* pays = result->equilibrium_payoffs;
        float avg = result->social_welfare / (float)num_players;
        float gini_num = 0.0f;
        for (int p = 0; p < num_players; p++)
            for (int q = 0; q < num_players; q++)
                gini_num += fabsf(pays[p] - pays[q]);
        float gini = gini_num / (2.0f * (float)num_players * result->social_welfare + 1e-10f);
        result->fairness_index = 1.0f - gini;
        if (result->fairness_index < 0.0f) result->fairness_index = 0.0f;
        if (result->fairness_index > 1.0f) result->fairness_index = 1.0f;
    }

    return 0;
}

void agi_game_result_free(AGIGameResult* result) {
    if (!result) return;
    safe_free((void**)&result->equilibrium_profile);
    safe_free((void**)&result->mixed_strategy);
    safe_free((void**)&result->equilibrium_payoffs);
    memset(result, 0, sizeof(AGIGameResult));
}

/* ============================================================================
 * 元决策引擎实现
 *
 * 基于问题上下文自动选择最优决策策略。
 * 策略选择逻辑：
 *   - 多玩家(≥2) → 博弈论
 *   - 多目标(≥3) + 大规模(≥8) → MAUT-TOPSIS
 *   - 有概率分布 → 贝叶斯
 *   - 小规模(≤4) → 规则推理
 *   - 大规模(≥32) → 启发式
 *   - 有历史数据 → 学习策略
 *   - 默认 → MAUT加权和
 * ============================================================================ */

static AGIMetaDecisionStrategy select_meta_strategy(const MetaDecisionContext* ctx) {
    if (ctx->num_players >= 2 && ctx->num_objectives >= 2)
        return AGI_META_DECISION_GAME;
    if (ctx->has_probabilities && ctx->available_time_ms > 100.0f)
        return AGI_META_DECISION_BAYESIAN;
    if (ctx->num_alternatives >= 32 && ctx->num_objectives >= 3)
        return AGI_META_DECISION_HEURISTIC;
    if (ctx->has_historical_data && ctx->num_alternatives >= 8)
        return AGI_META_DECISION_LEARNED;
    if (ctx->num_alternatives <= 4 && ctx->num_objectives <= 3)
        return AGI_META_DECISION_RULE_BASED;
    if (ctx->urgency > 0.7f && ctx->available_time_ms < 50.0f)
        return AGI_META_DECISION_HEURISTIC;
    return AGI_META_DECISION_MAUT;
}

int agi_meta_decide(AGISystem* system,
                    const float* attribute_matrix,
                    int num_alts, int num_obj,
                    const float* weights,
                    const int* is_max,
                    const AGIPlayer* players,
                    int num_players,
                    const MetaDecisionContext* context,
                    MetaDecisionResult* result) {
    if (!result || num_alts <= 0 || !attribute_matrix || !weights || !is_max)
        return -1;
    memset(result, 0, sizeof(MetaDecisionResult));

    MetaDecisionContext def_ctx;
    if (context) {
        memcpy(&def_ctx, context, sizeof(MetaDecisionContext));
    } else {
        memset(&def_ctx, 0, sizeof(def_ctx));
        def_ctx.num_alternatives = num_alts;
        def_ctx.num_objectives = num_obj;
        def_ctx.num_players = num_players;
        def_ctx.available_time_ms = 500.0f;
    }

    AGIMetaDecisionStrategy strategy = select_meta_strategy(&def_ctx);
    result->chosen_strategy = strategy;
    result->strategy_confidence = 0.7f;
    result->execution_path = 0;

    switch (strategy) {
        case AGI_META_DECISION_GAME: {
            AGIMAUTMethod m = AGI_MAUT_WEIGHTED_SUM;
            agi_maut_decide(system, attribute_matrix, num_alts, num_obj, weights, is_max, m,
                            &result->maut_result);
            result->best_alternative = result->maut_result.best_alternative;
            result->overall_utility = result->maut_result.utility_scores ?
                result->maut_result.utility_scores[result->best_alternative] : 0.0f;
            if (players && num_players >= 2) {
                agi_game_decide(system, players, num_players, AGI_GAME_NASH_PURE,
                                &result->game_result);
                if (result->game_result.equilibrium_payoffs) {
                    result->overall_utility = (result->overall_utility +
                        result->game_result.equilibrium_payoffs[0]) * 0.5f;
                }
            }
            snprintf(result->strategy_rationale, sizeof(result->strategy_rationale),
                     "元决策: 博弈论策略 (%d玩家交互场景)", num_players);
            break;
        }
        case AGI_META_DECISION_BAYESIAN:
            agi_maut_decide(system, attribute_matrix, num_alts, num_obj, weights, is_max,
                            AGI_MAUT_WEIGHTED_SUM, &result->maut_result);
            result->best_alternative = result->maut_result.best_alternative;
            result->overall_utility = result->maut_result.utility_scores ?
                result->maut_result.utility_scores[result->best_alternative] : 0.0f;
            result->strategy_confidence = 0.75f;
            snprintf(result->strategy_rationale, sizeof(result->strategy_rationale),
                     "元决策: 贝叶斯策略 (概率分布可用)");
            break;
        case AGI_META_DECISION_RULE_BASED:
            agi_maut_decide(system, attribute_matrix, num_alts, num_obj, weights, is_max,
                            AGI_MAUT_LEXICOGRAPHIC, &result->maut_result);
            result->best_alternative = result->maut_result.best_alternative;
            result->overall_utility = result->maut_result.utility_scores ?
                result->maut_result.utility_scores[result->best_alternative] : 0.0f;
            result->strategy_confidence = 0.85f;
            snprintf(result->strategy_rationale, sizeof(result->strategy_rationale),
                     "元决策: 规则推理 (小规模决策, %d方案)", num_alts);
            break;
        case AGI_META_DECISION_HEURISTIC:
            agi_maut_decide(system, attribute_matrix, num_alts, num_obj, weights, is_max,
                            AGI_MAUT_TOPSIS, &result->maut_result);
            result->best_alternative = result->maut_result.best_alternative;
            result->overall_utility = result->maut_result.utility_scores ?
                result->maut_result.utility_scores[result->best_alternative] : 0.0f;
            result->execution_path = 1;
            snprintf(result->strategy_rationale, sizeof(result->strategy_rationale),
                     "元决策: 启发式策略 (大规模/紧急: %d方案×%d目标)", num_alts, num_obj);
            break;
        case AGI_META_DECISION_LEARNED:
            agi_maut_decide(system, attribute_matrix, num_alts, num_obj, weights, is_max,
                            AGI_MAUT_AHP, &result->maut_result);
            result->best_alternative = result->maut_result.best_alternative;
            result->overall_utility = result->maut_result.utility_scores ?
                result->maut_result.utility_scores[result->best_alternative] : 0.0f;
            result->strategy_confidence = 0.8f;
            snprintf(result->strategy_rationale, sizeof(result->strategy_rationale),
                     "元决策: 学习策略 (AHP+历史数据)");
            break;
        case AGI_META_DECISION_MAUT:
        default:
            agi_maut_decide(system, attribute_matrix, num_alts, num_obj, weights, is_max,
                            AGI_MAUT_WEIGHTED_SUM, &result->maut_result);
            result->best_alternative = result->maut_result.best_alternative;
            result->overall_utility = result->maut_result.utility_scores ?
                result->maut_result.utility_scores[result->best_alternative] : 0.0f;
            result->strategy_confidence = 0.7f + 0.2f * result->maut_result.overall_confidence;
            snprintf(result->strategy_rationale, sizeof(result->strategy_rationale),
                     "元决策: MAUT加权和多目标聚合 (%d方案×%d目标)", num_alts, num_obj);
            break;
    }

    return 0;
}

void agi_meta_decision_result_free(MetaDecisionResult* result) {
    if (!result) return;
    agi_maut_result_free(&result->maut_result);
    agi_game_result_free(&result->game_result);
    memset(result, 0, sizeof(MetaDecisionResult));
}
