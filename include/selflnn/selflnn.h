/**
 * SELF-LNN 主头文件
 * 
 * 包含所有SELF-LNN模块的公共API接口。
 * 用户只需包含此头文件即可使用所有SELF-LNN功能。
 *
 * API分组（按功能域）：
 *   1. 初始化/销毁 —— 系统生命周期管理
 *   2. 系统状态   —— 运行时信息查询
 *   3. 配置管理   —— 配置文件加载/保存/运行时修改
 *   4. 子系统获取 —— 全局单例子系统访问器
 *   5. 单一LNN模型管理 —— 全模态共享同一个连续动态系统
 *   6. 训练与检查点 —— 模型训练/加载/保存
 *   7. 知识推理→LNN连接通道
 *   8. AGI后台任务状态访问器
 *   9. 高级功能API —— 示例程序和集成使用
 */

#ifndef SELFLNN_H
#define SELFLNN_H

#include "selflnn/core/common.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/cfc.h"

/* 标准库：Clang 要求显式声明 time_t */
#include <time.h>

/* 补充核心液态神经网络模块头文件 */
#include "selflnn/core/cfc_network.h"
#include "selflnn/core/cfc_enhanced.h"
#include "selflnn/core/cma_es.h"
#include "selflnn/core/decision_engine.h"
#include "selflnn/core/dynamics.h"
#include "selflnn/core/evolutionary_algorithms.h"
#include "selflnn/core/graph_optimization.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/lnn_layer_norm.h"
#include "selflnn/core/loss.h"
#include "selflnn/core/ode_solvers.h"
#include "selflnn/core/optimizer.h"
#include "selflnn/core/quaternion_cfc.h"
#include "selflnn/core/quaternion_lnn.h"
#include "selflnn/core/quaternion_conv.h"
#include "selflnn/core/quaternion_optimizer.h"
#include "selflnn/core/quaternion_enhanced.h"
#include "selflnn/core/quaternion_batchnorm.h"
#include "selflnn/core/laplace.h"
#include "selflnn/core/laplace_unified.h"
#include "selflnn/core/laplace_features.h"
#include "selflnn/core/state.h"
#include "selflnn/core/tensor.h"
#include "selflnn/core/unified_lnn_state.h"
#include "selflnn/core/system_scheduler.h"
#include "selflnn/core/architecture_controller.h" /* P0-001: 动态架构控制器 */

/* 补充认知模块头文件 */
#include "selflnn/cognition/self_cognition.h"
#include "selflnn/cognition/metacognition.h"
#include "selflnn/cognition/deep_correction.h"
#include "selflnn/cognition/deep_reflection.h"
#include "selflnn/cognition/deep_thought_chain.h"
#include "selflnn/cognition/abstraction.h"
#include "selflnn/agi/agi.h"
#include "selflnn/agi/capability_switch.h"
#include "selflnn/agi/task_scheduler.h"

// 多模态处理
#include "selflnn/multimodal/unified_signal_processor.h"
#include "selflnn/multimodal/liquid_vision.h"
#include "selflnn/multimodal/audio.h"
#include "selflnn/multimodal/audio_tools.h"
#include "selflnn/multimodal/text.h"
/* 补充多模态模块头文件 */
#include "selflnn/multimodal/speech_recognition.h"
#include "selflnn/multimodal/tts.h"
#include "selflnn/multimodal/vad.h"
#include "selflnn/multimodal/dialogue.h"
#include "selflnn/multimodal/dialogue_memory.h"
#include "selflnn/multimodal/speech_language_model.h"
#include "selflnn/multimodal/slam.h"
#include "selflnn/multimodal/slam_enhance.h"
#include "selflnn/multimodal/depth_estimation.h"
#include "selflnn/multimodal/stereo_calibration.h"
#include "selflnn/multimodal/stereo_depth_enhance.h"
#include "selflnn/multimodal/stereo_3d_reconstruction.h"
#include "selflnn/multimodal/ocr.h"
#include "selflnn/multimodal/object_recognition.h"
#include "selflnn/multimodal/image_recognition_deep.h"
#include "selflnn/multimodal/multimodal_unified_input.h"
#include "selflnn/multimodal/multimodal_integration.h"
#include "selflnn/multimodal/multimodal_manager.h"
#include "selflnn/multimodal/multimodal_teaching.h"
#include "selflnn/multimodal/teaching_loop.h"
#include "selflnn/multimodal/sensor.h"
#include "selflnn/multimodal/sensor_preprocessor_deep.h"
#include "selflnn/multimodal/haptic_enhance.h"
#include "selflnn/multimodal/haptic_learning.h"
#include "selflnn/multimodal/data_collection_pipeline.h"
#include "selflnn/multimodal/camera_capture.h"
/* audio_capture_* 函数声明已在 audio.h (L208-287) 中提供，无需重复包含 */
#include "selflnn/multimodal/image_loader.h"
#include "selflnn/multimodal/audio_loader.h"
#include "selflnn/multimodal/character_segmentation.h"
#include "selflnn/multimodal/text_detection.h"
#include "selflnn/multimodal/point_cloud.h"

// 认知功能
#include "selflnn/reasoning/reasoning.h"
#include "selflnn/reasoning/planning.h"
#include "selflnn/reasoning/long_term_planning.h"
#include "selflnn/reasoning/math_physics_reasoning.h"
#include "selflnn/reasoning/causal_reasoning.h"
#include "selflnn/reasoning/hierarchical_planning.h"
#include "selflnn/reasoning/planning_enhanced.h"
#include "selflnn/reasoning/cfc_uncertainty_reasoning.h"
#include "selflnn/reasoning/quaternion_liquid_gate.h"
#include "selflnn/reasoning/laplace_operator.h"

// 知识库
#include "selflnn/knowledge/knowledge_graph.h"
#include "selflnn/knowledge/semantic_network.h"
#include "selflnn/knowledge/logic_reasoning.h"
#include "selflnn/knowledge/knowledge_integration.h"
#include "selflnn/knowledge/knowledge_inference.h"
#include "selflnn/knowledge/knowledge_version.h"
#include "selflnn/knowledge/cfc_knowledge_embedding.h"
#include "selflnn/knowledge/graph_query.h"
#include "selflnn/knowledge/graph_reasoning.h"
#include "selflnn/knowledge/graph_storage.h"
#include "selflnn/knowledge/knowledge_self_check.h"
#include "selflnn/knowledge/ontology_engineering.h"
#include "selflnn/knowledge/semantic_parsing.h"
#include "selflnn/knowledge/uncertainty_reasoning.h"
#include "selflnn/knowledge/auto_learning.h"
#include "selflnn/knowledge/api_training.h"
#include "selflnn/knowledge/skill_library.h"

// 记忆系统
#include "selflnn/memory/memory_manager.h"
#include "selflnn/memory/short_term.h"
#include "selflnn/memory/long_term.h"
#include "selflnn/memory/episodic.h"
#include "selflnn/memory/semantic.h"
#include "selflnn/memory/working_memory.h"

// 集成孤儿子多模态传感器模块
#include "selflnn/multimodal/motor.h"
#include "selflnn/multimodal/radar.h"
#include "selflnn/multimodal/thermal.h"
#include "selflnn/multimodal/proprioception.h"
#include "selflnn/multimodal/environment_sound.h"

// 应用能力
#include "selflnn/programming/self_programming.h"
#include "selflnn/programming/programming_enhanced.h"  /* M-027修复 */
#include "selflnn/programming/c_interpreter.h"         /* M-027修复 */
#include "selflnn/product_design/product_design.h"
#include "selflnn/product_design/product_design_enhanced.h"
#include "selflnn/multisystem/multisystem_control.h"
#include "selflnn/robot/hardware_interface.h"
#include "selflnn/robot/simulator.h"
#include "selflnn/robot/dynamics.h"
#include "selflnn/robot/swarm_coordination.h"
#include "selflnn/robot/physics_engine.h"
#include "selflnn/robot/robot.h"

// 基础设施
#include "selflnn/gpu/gpu.h"
#include "selflnn/concurrency/thread_pool.h"

/* 补充学习、演化、训练、安全、分布式模块头文件 */
#include "selflnn/learning/learning.h"
#include "selflnn/learning/imitation_learning.h"
#include "selflnn/learning/imitation_deep.h"
#include "selflnn/learning/meta_learning.h"
#include "selflnn/learning/exploration_strategies.h"
#include "selflnn/learning/online_learning.h"
#include "selflnn/learning/reinforcement_learning.h"
#include "selflnn/learning/multi_agent.h"
#include "selflnn/learning/teach_by_showing.h"
#include "selflnn/learning/manual_learning.h"

#include "selflnn/evolution/evolution_engine.h"
#include "selflnn/evolution/neural_architecture_search.h"
#include "selflnn/evolution/pareto_optimization.h"

#include "selflnn/training/training.h"
#include "selflnn/training/training_pipeline.h"
#include "selflnn/training/training_enhanced.h"
#include "selflnn/training/model_parallel.h"
#include "selflnn/training/mixed_precision.h"
#include "selflnn/training/data_loaders.h"
#include "selflnn/training/model_registry.h"
#include "selflnn/training/model_version.h"
#include "selflnn/training/regularization.h"
#include "selflnn/training/training_monitor.h"
#include "selflnn/training/training_data_pipeline.h"
#include "selflnn/training/training_dataset.h"
#include "selflnn/training/distributed_training.h"

#include "selflnn/safety/safety_monitor.h"
#include "selflnn/safety/emergency_stop.h"
#include "selflnn/safety/content_filter.h"
#include "selflnn/safety/audit_logger.h"
#include "selflnn/safety/security_monitor_deep.h"

#include "selflnn/distributed/load_balancer.h"
#include "selflnn/distributed/pbft.h"

#include "selflnn/backend/backend.h"
#include "selflnn/backend/auth.h"
#include "selflnn/backend/websocket_push.h"

// 实用工具
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/perf.h"
#include "selflnn/utils/energy_efficiency.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * 类型定义
 * ================================================================ */

/* L-004修复: 不透明指针类型声明 —— 替代 void*，提供编译期类型安全
 * 每个类型对应一个子系统单例，外部仅持有不透明指针，
 * 内部结构体定义在各自模块的 .c 文件中通过 SELFLNN_IMPLEMENTATION 暴露。
 * 多重前向声明在C11中合法（6.7.2.3p5），与子头文件中的声明兼容。 */
typedef struct SelfCognition        SelfCognition;
typedef struct MetaCognition        MetaCognition;
typedef struct KnowledgeGraph       KnowledgeGraph;
typedef struct GraphReasoner        GraphReasoner;
typedef struct GpuContext           GpuContext;
typedef struct DialogueProcessor    DialogueProcessor;
typedef struct DialogueMemory       DialogueMemory;
typedef struct PlanningSystem       PlanningSystem;
typedef struct OnlineLearner        OnlineLearner;
typedef struct MultiAgentSystem     MultiAgentSystem;
typedef struct EvolutionEngine      EvolutionEngine;
typedef struct AutoLearning         AutoLearning;
typedef struct ReasoningEngine      ReasoningEngine;
typedef struct CausalReasoningEngine CausalReasoningEngine;
typedef struct MemoryManager        MemoryManager;
typedef struct KnowledgeBase        KnowledgeBase;
typedef struct KnowledgeInference   KnowledgeInference;
typedef struct ThreadPool           ThreadPool;
typedef struct SafetyMonitor        SafetyMonitor;
typedef struct DataPipeline         DataPipeline;
typedef struct SpeechRecognizer     SpeechRecognizer;
typedef struct ProductDesignEngine  ProductDesignEngine;
typedef struct MultiSystemControl   MultiSystemControl;
typedef struct SelfProgrammingEngine SelfProgrammingEngine;
typedef struct DistributedContext   DistributedContext;
typedef struct NasSystem            NasSystem;
typedef struct LaplaceUnified       LaplaceUnified;
typedef struct AudioCapture         AudioCapture;
typedef struct TtsEngine            TtsEngine;
typedef struct ComputerOperation    ComputerOperation;
typedef struct AuditLogger          AuditLogger;
typedef struct ContentFilter        ContentFilter;
typedef struct LoadBalancer         LoadBalancer;
typedef struct RwLockMap            RwLockMap;
typedef struct PbftSystem           PbftSystem;
typedef struct TrainingPipeline     TrainingPipeline;
typedef struct SecurityMonitorDeep  SecurityMonitorDeep;
typedef struct TeachingLoop         TeachingLoop;
typedef struct MultimodalTeaching   MultimodalTeaching;
typedef struct UnifiedSignalProcessor UnifiedSignalProcessor;
typedef struct UnifiedSignalProcessorAdvanced UnifiedSignalProcessorAdvanced;
typedef struct UnifiedSignalProcessorTraining UnifiedSignalProcessorTraining;
typedef struct TeachingSystem       TeachingSystem;
typedef struct UnifiedState         UnifiedState;  /* 统一状态（与UnifiedLNNState互补） */
/* LNN/UnifiedLNNState 已在 lnn.h / unified_lnn_state.h 中声明 */

typedef struct {
    int state_dimension;
    int multimodal_channels;
    int memory_capacity;
    int max_concurrent_tasks;
    PowerMode power_mode;
    GpuBackend gpu_backend;
    /**
     * @brief 模型文件路径（悬空指针）
     * 
     * 【生命周期要求】此指针仅在 selflnn_init() 调用期间被读取，
     * 用于自动加载模型检查点。系统内部不会复制此字符串，
     * 仅存储指针引用。调用栈变量在 init 期间有效即可。
     * 初始化完成后此字段不再被访问，无需长期保持有效。
     */
    const char* model_path;
/* 新增端口字段 — 原只在保存时写端口但从不加载 */
    int http_port;
    int websocket_port;
    int distributed_port;
    int mixed_precision_mode;  /* M-035: 混合精度模式 (0=关闭, 1=auto, 2=FP16, 3=BF16) */
} SystemConfig;

typedef struct {
    double* state_vector;
    size_t state_dimension;
    double timestamp;
    int cognitive_load;
    double confidence;
} SystemState;

typedef struct {
    void* vision_data;
    void* audio_data;
    void* text_data;
    void* sensor_data;
    void* control_data;
    double timestamp;
} MultimodalInput;

typedef struct {
    int major;
    int minor;
    int patch;
    const char* build_time;
    const char* git_commit;
} VersionInfo;

typedef struct {
    double uptime;
    double memory_usage_mb;
    double cpu_usage_percent;
    int active_tasks;
    int total_memories;
    int total_knowledge;
    int hardware_available;
    double real_memory_total_mb;
    double real_memory_free_mb;
    double real_cpu_count;
} SystemStatus;

/* ---- 示例程序所需类型 ---- */

typedef enum {
    LANGUAGE_C = 0,
    LANGUAGE_CPP = 1,
    LANGUAGE_PYTHON = 2,
    LANGUAGE_JAVA = 3
} Language;

typedef struct {
    int complexity_score;
    double analysis_time;
    size_t issue_count;
    char** issues;
    size_t suggestion_count;
    char** suggestions;
} CodeAnalysis;

typedef struct {
    double timestamp;
    double power_watt;
    double energy_joule;
    double temperature_c;
} EnergyDataPoint;

/**
 * @brief 产品设计标签数据结构
 * 
 * 【数组大小说明】以下数组大小与对应的计数字段存在严格对应关系：
 * - type_labels[4]     <-> 无计数字段（固定4种类型标签）
 * - style_suffixes[5]  <-> style_suffix_count（样式后缀实际数量，<=5）
 * - feat_prefixes[16]  <-> feat_prefix_count（特征前缀实际数量，<=16）
 * - feat_suffixes[12]  <-> feat_suffix_count（特征后缀实际数量，<=12）
 * - default_features[3] <-> default_feature_count（默认特征实际数量，<=3）
 * 
 * 访问时必须以对应计数字段为准，不得超过数组边界。
 * 所有数组大小均为编译期常量，不可动态调整。
 */
typedef struct {
    const char* type_labels[4];
    const char* style_suffixes[5];
    size_t style_suffix_count;
    const char* feat_prefixes[16];
    size_t feat_prefix_count;
    const char* feat_suffixes[12];
    size_t feat_suffix_count;
    const char* default_features[3];
    size_t default_feature_count;
} ProductDesignLabels;

/* ================================================================
 * 1. 初始化/销毁 —— 系统生命周期管理
 * ================================================================ */

SELFLNN_API int selflnn_init(const SystemConfig* config);
SELFLNN_API int selflnn_shutdown(void);
SELFLNN_API void selflnn_module_init(void);
SELFLNN_API void selflnn_module_cleanup(void);

/* ================================================================
 * 2. 系统状态 —— 运行时信息查询
 * ================================================================ */

SELFLNN_API int selflnn_process_input(const MultimodalInput* input, SystemState* state);
SELFLNN_API int selflnn_get_version(VersionInfo* version);
SELFLNN_API int selflnn_get_status(SystemStatus* status);
SELFLNN_API int selflnn_get_last_error(void);
SELFLNN_API const char* selflnn_get_error_message(int error_code);

/* ================================================================
 * 3. 配置管理 —— 配置文件加载/保存/运行时修改
 * ================================================================ */

SELFLNN_API int selflnn_config_load_from_file(const char* filepath, SystemConfig* config);
SELFLNN_API int selflnn_config_save_to_file(const char* filepath, const SystemConfig* config);
/* 配置Schema验证 */
SELFLNN_API int selflnn_config_validate(const SystemConfig* config, char* error_msg, size_t msg_size);
SELFLNN_API int selflnn_set_power_mode(PowerMode power_mode);
SELFLNN_API void selflnn_set_product_design_labels(const ProductDesignLabels* labels);
SELFLNN_API const ProductDesignLabels* selflnn_get_product_design_labels(void);

/* ================================================================
 * 4. 子系统获取 —— 全局单例子系统访问器
 *    所有子系统通过以下函数获取共享实例，禁止自行创建。
 * ================================================================ */

/* ---- 4a. 核心认知模块 ---- */
SELFLNN_API SelfCognition*    selflnn_get_self_cognition(void);
SELFLNN_API MetaCognition*    selflnn_get_metacognition(void);
SELFLNN_API KnowledgeGraph*   selflnn_get_knowledge_graph(void);
SELFLNN_API GraphReasoner*    selflnn_get_graph_reasoner(void);
SELFLNN_API GpuContext*       selflnn_get_gpu_context(void);
SELFLNN_API DialogueProcessor* selflnn_get_dialogue_processor(void);
SELFLNN_API DialogueMemory*   selflnn_get_dialogue_memory(void);
SELFLNN_API PlanningSystem*   selflnn_get_planning_system(void);

/* ---- 4b. 学习与演化模块 ---- */
SELFLNN_API OnlineLearner*    selflnn_get_online_learner(void);
SELFLNN_API MultiAgentSystem* selflnn_get_multi_agent_system(void);
SELFLNN_API EvolutionEngine*  selflnn_get_evolution_engine(void);
SELFLNN_API AutoLearning*     selflnn_get_auto_learning(void);

/* ---- 4c. 推理与知识模块 ---- */
SELFLNN_API ReasoningEngine*       selflnn_get_reasoning_engine(void);
SELFLNN_API CausalReasoningEngine* selflnn_get_causal_reasoning_engine(void);
SELFLNN_API MemoryManager*         selflnn_get_memory_manager(void);
SELFLNN_API KnowledgeBase*         selflnn_get_knowledge_base(void);
SELFLNN_API KnowledgeInference*    selflnn_get_knowledge_inference(void);

/* ---- 4d. 基础设施模块 ---- */
SELFLNN_API ThreadPool*            selflnn_get_thread_pool(void);
SELFLNN_API SafetyMonitor*         selflnn_get_safety_monitor(void);
SELFLNN_API DataPipeline*          selflnn_get_data_pipeline(void);
SELFLNN_API SpeechRecognizer*      selflnn_get_speech_recognizer(void);
SELFLNN_API void                   selflnn_set_speech_recognizer(SpeechRecognizer* sr);
SELFLNN_API ProductDesignEngine*   selflnn_get_product_design_engine(void);
SELFLNN_API void                   selflnn_set_product_design_engine(ProductDesignEngine* engine);
SELFLNN_API MultiSystemControl*    selflnn_get_multisystem_control(void);
SELFLNN_API SelfProgrammingEngine* selflnn_get_self_programming_engine(void);
SELFLNN_API DistributedContext*    selflnn_get_distributed_context(void);
SELFLNN_API NasSystem*             selflnn_get_nas_system(void);
SELFLNN_API LaplaceUnified*        selflnn_get_laplace_unified(void);
SELFLNN_API AudioCapture*          selflnn_get_audio_capture(void);
SELFLNN_API TtsEngine*             selflnn_get_tts_engine(void);
SELFLNN_API ComputerOperation*     selflnn_get_computer_operation(void);
SELFLNN_API AuditLogger*           selflnn_get_audit_logger(void);
SELFLNN_API ContentFilter*         selflnn_get_content_filter(void);
SELFLNN_API LoadBalancer*          selflnn_get_load_balancer(void);
SELFLNN_API RwLockMap*             selflnn_get_rw_lock_map(void);
SELFLNN_API PbftSystem*            selflnn_get_pbft_system(void);
SELFLNN_API TrainingPipeline*      selflnn_get_training_pipeline(void);
SELFLNN_API void                   selflnn_set_training_pipeline(TrainingPipeline* pipeline);
SELFLNN_API size_t                 selflnn_get_config_state_dimension(void);
SELFLNN_API void                   selflnn_set_laplace_metrics(const float* metrics, int count);
SELFLNN_API SecurityMonitorDeep*   selflnn_get_security_monitor_deep(void);
SELFLNN_API TeachingLoop*          selflnn_get_teaching_loop(void);
SELFLNN_API MultimodalTeaching*    selflnn_get_multimodal_teaching(void);

/* ---- 4e. 事件驱动即时自检 ---- */
SELFLNN_API void dcpipeline_request_immediate_check(void);
SELFLNN_API int  dcpipeline_is_immediate_check_requested(void);
SELFLNN_API void dcpipeline_clear_immediate_check(void);

/* ---- 4f. 多模态处理模块 ---- */
SELFLNN_API UnifiedSignalProcessor*          selflnn_get_unified_signal_processor(void);
SELFLNN_API UnifiedSignalProcessorAdvanced*  selflnn_get_unified_signal_processor_advanced(void);
SELFLNN_API UnifiedSignalProcessorTraining*  selflnn_get_unified_signal_processor_training(void);

/* P2-007修复: selflnn_get_teaching_system的显式声明 */
SELFLNN_API TeachingSystem* selflnn_get_teaching_system(void);

/* ================================================================
 * 5. 单一LNN模型管理 —— 全模态共享同一个连续动态系统
 *    整个系统只允许存在一个LNN实例，所有模块共享。
 *    任何模块禁止自行调用 lnn_create 创建独立LNN。
 * ================================================================ */

SELFLNN_API LNN* selflnn_get_lnn(void);
SELFLNN_API LNN* selflnn_get_shared_lnn(void);
SELFLNN_API LNN* selflnn_get_or_create_lnn(const LNNConfig* config);
SELFLNN_API UnifiedLNNState* selflnn_get_unified_lnn_state(void);
SELFLNN_API UnifiedState*    selflnn_get_unified_state(void);
SELFLNN_API void selflnn_enforce_single_lnn(void);
SELFLNN_API int selflnn_is_single_lnn_enforced(void);
SELFLNN_API int selflnn_register_module(int module_id, void* instance, int uses_shared_lnn);
SELFLNN_API int selflnn_module_uses_shared_lnn(int module_id);

/* LNN并发安全：带锁保护的前向传播 */
SELFLNN_API int selflnn_safe_forward(LNN* lnn, const float* input, float* output);

/* ================================================================
 * 6. 训练与检查点 —— 模型训练/加载/保存
 * ================================================================ */

SELFLNN_API int selflnn_checkpoints_auto_load(void);
SELFLNN_API int selflnn_save_checkpoint(const char* filepath); /* 检查点保存 */

/* 引导多模态模块训练状态，检查点加载后调用 */
SELFLNN_API void selflnn_bootstrap_trained_modules(void);

/* ================================================================
 * 7. 知识推理→LNN连接通道
 *    将知识推理增强引擎的符号化推理结果映射为LNN状态扰动，
 *    实现"知识推理→LNN→决策"的完整数据通路。
 * ================================================================ */

SELFLNN_API int selflnn_consume_knowledge_inference(void* lnn_instance, void* kie,
                                                     const char* query_concept,
                                                     int max_hops,
                                                     float perturbation_strength);

/* 知识库更新事件通知机制
 * selflnn_trigger_knowledge_refresh: 知识库回调调用，设置刷新标志
 * selflnn_check_and_reset_knowledge_refresh: AGI后台循环查询并重置标志 */
SELFLNN_API void selflnn_trigger_knowledge_refresh(void);
SELFLNN_API int selflnn_check_and_reset_knowledge_refresh(void);

/* ================================================================
 * 8. AGI后台任务所需的状态访问器
 *    供main.c后台任务读取LNN状态和知识库信息。
 * ================================================================ */

SELFLNN_API int selflnn_get_recent_state(void* lnn, float* state, int dim);
SELFLNN_API int selflnn_get_recent_output(void* lnn, float* output, int dim);
SELFLNN_API int selflnn_get_active_goal(void* kb, float* goal, int dim);

/* ================================================================
 * 9. 高级功能API —— 示例程序和集成使用
 * ================================================================ */

SELFLNN_API int selflnn_analyze_code(const char* source_code, Language language, CodeAnalysis* analysis);
SELFLNN_API int selflnn_design_product(const ProductRequirement* requirement, ProductSpec* design);
SELFLNN_API size_t selflnn_discover_devices(DeviceInfo* devices, size_t max_devices);
SELFLNN_API size_t selflnn_monitor_energy(double duration, EnergyDataPoint* data_points, size_t max_points);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_H
