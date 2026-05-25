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

// 多模态处理
#include "selflnn/multimodal/unified_signal_processor.h"
#include "selflnn/multimodal/liquid_vision.h"
#include "selflnn/multimodal/audio.h"
#include "selflnn/multimodal/text.h"

// 认知功能
#include "selflnn/reasoning/reasoning.h"
#include "selflnn/reasoning/planning.h"
#include "selflnn/reasoning/long_term_planning.h"
#include "selflnn/reasoning/math_physics_reasoning.h"

// 知识库
#include "selflnn/knowledge/knowledge_graph.h"
#include "selflnn/knowledge/semantic_network.h"
#include "selflnn/knowledge/logic_reasoning.h"
#include "selflnn/knowledge/knowledge_integration.h"

// 记忆系统
#include "selflnn/memory/memory_manager.h"
#include "selflnn/memory/short_term.h"
#include "selflnn/memory/long_term.h"
#include "selflnn/memory/episodic.h"
#include "selflnn/memory/semantic.h"

// 应用能力
#include "selflnn/programming/self_programming.h"
#include "selflnn/product_design/product_design.h"
#include "selflnn/multisystem/multisystem_control.h"
#include "selflnn/robot/hardware_interface.h"
#include "selflnn/robot/simulator.h"

// 基础设施
#include "selflnn/gpu/gpu.h"
#include "selflnn/concurrency/thread_pool.h"

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

typedef struct {
    int state_dimension;
    int multimodal_channels;
    int memory_capacity;
    int max_concurrent_tasks;
    PowerMode power_mode;
    GpuBackend gpu_backend;
    const char* model_path;
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
SELFLNN_API int selflnn_set_power_mode(PowerMode power_mode);
SELFLNN_API void selflnn_set_product_design_labels(const ProductDesignLabels* labels);
SELFLNN_API const ProductDesignLabels* selflnn_get_product_design_labels(void);

/* ================================================================
 * 4. 子系统获取 —— 全局单例子系统访问器
 *    所有子系统通过以下函数获取共享实例，禁止自行创建。
 * ================================================================ */

/* ---- 4a. 核心认知模块 ---- */
SELFLNN_API void* selflnn_get_self_cognition(void);
SELFLNN_API void* selflnn_get_metacognition(void);
SELFLNN_API void* selflnn_get_dialogue_processor(void);
SELFLNN_API void* selflnn_get_planning_system(void);

/* ---- 4b. 学习与演化模块 ---- */
SELFLNN_API void* selflnn_get_online_learner(void);
SELFLNN_API void* selflnn_get_multi_agent_system(void); /* H-015集成 */
SELFLNN_API void* selflnn_get_evolution_engine(void);
SELFLNN_API void* selflnn_get_auto_learning(void);

/* ---- 4c. 推理与知识模块 ---- */
SELFLNN_API void* selflnn_get_reasoning_engine(void);
SELFLNN_API void* selflnn_get_memory_manager(void);
SELFLNN_API void* selflnn_get_knowledge_base(void);
SELFLNN_API void* selflnn_get_knowledge_inference(void);

/* ---- 4d. 基础设施模块 ---- */
SELFLNN_API void* selflnn_get_thread_pool(void);
SELFLNN_API void* selflnn_get_safety_monitor(void);
SELFLNN_API void* selflnn_get_data_pipeline(void);
SELFLNN_API void* selflnn_get_speech_recognizer(void);
SELFLNN_API void selflnn_set_speech_recognizer(void* sr);
SELFLNN_API void* selflnn_get_product_design_engine(void);  /* APP10: 产品设计引擎 */
SELFLNN_API void selflnn_set_product_design_engine(void* engine);
SELFLNN_API void* selflnn_get_multisystem_control(void);     /* APP13: 多系统控制 */

/* ---- 4e. 事件驱动即时自检（ZSFWS-038） ---- */
SELFLNN_API void dcpipeline_request_immediate_check(void);
SELFLNN_API int  dcpipeline_is_immediate_check_requested(void);
SELFLNN_API void dcpipeline_clear_immediate_check(void);

/* ---- 4f. 多模态处理模块 ---- */
SELFLNN_API void* selflnn_get_unified_signal_processor(void);

/* ================================================================
 * 5. 单一LNN模型管理 —— 全模态共享同一个连续动态系统
 *    整个系统只允许存在一个LNN实例，所有模块共享。
 *    任何模块禁止自行调用 lnn_create() 创建独立LNN。
 * ================================================================ */

SELFLNN_API void* selflnn_get_lnn(void);
SELFLNN_API void* selflnn_get_shared_lnn(void);
SELFLNN_API LNN* selflnn_get_or_create_lnn(const LNNConfig* config);
SELFLNN_API void* selflnn_get_unified_lnn_state(void);
SELFLNN_API void* selflnn_get_unified_state(void);
SELFLNN_API void selflnn_enforce_single_lnn(void);
SELFLNN_API int selflnn_is_single_lnn_enforced(void);
SELFLNN_API int selflnn_register_module(int module_id, void* instance, int uses_shared_lnn);
SELFLNN_API int selflnn_module_uses_shared_lnn(int module_id);

/* ================================================================
 * 6. 训练与检查点 —— 模型训练/加载/保存
 * ================================================================ */

SELFLNN_API int selflnn_checkpoints_auto_load(void);

/* ================================================================
 * 7. 知识推理→LNN连接通道
 *    将知识推理增强引擎的符号化推理结果映射为LNN状态扰动，
 *    实现"知识推理→LNN→决策"的完整数据通路。
 * ================================================================ */

SELFLNN_API int selflnn_consume_knowledge_inference(void* lnn_instance, void* kie,
                                                     const char* query_concept,
                                                     int max_hops,
                                                     float perturbation_strength);

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
