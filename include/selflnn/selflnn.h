/**
 * SELF-LNN 主头文件
 * 
 * 包含所有SELF-LNN模块的公共API接口。
 * 用户只需包含此头文件即可使用所有SELF-LNN功能。
 */

#ifndef SELFLNN_H
#define SELFLNN_H

#include "selflnn/core/common.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/cfc.h"

// 多模态处理
#include "selflnn/multimodal/unified_signal_processor.h"
#include "selflnn/multimodal/vision.h"
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

// 主系统API
#ifdef __cplusplus
extern "C" {
#endif

// 系统配置
typedef struct {
    int state_dimension;          // 状态维度
    int multimodal_channels;      // 多模态通道数
    int memory_capacity;         // 记忆容量
    int max_concurrent_tasks;    // 最大并发任务数
    PowerMode power_mode;        // 功率模式
    GpuBackend gpu_backend;      // GPU后端
    const char* model_path;      // 模型路径
} SystemConfig;

// 系统状态
typedef struct {
    double* state_vector;       // 状态向量
    size_t state_dimension;     // 状态维度
    double timestamp;           // 时间戳
    int cognitive_load;         // 认知负载
    double confidence;          // 置信度
} SystemState;

// 多模态输入
typedef struct {
    void* vision_data;          // 视觉数据
    void* audio_data;           // 音频数据
    void* text_data;            // 文本数据
    void* sensor_data;          // 传感器数据
    void* control_data;         // 控制信号
    double timestamp;           // 时间戳
} MultimodalInput;

// 版本信息
typedef struct {
    int major;                  // 主版本
    int minor;                  // 次版本
    int patch;                  // 修订版本
    const char* build_time;     // 构建时间
    const char* git_commit;     // Git提交哈希
} VersionInfo;

// 系统状态信息
typedef struct {
    double uptime;              // 运行时间（秒）
    double memory_usage_mb;     // 内存使用（MB）
    double cpu_usage_percent;   // CPU使用率（百分比）
    int active_tasks;           // 活动任务数
    int total_memories;         // 总记忆数
    int total_knowledge;        // 总知识数
    int hardware_available;     // 真实硬件数据是否可用（1=真实数据，0=硬件未连接）
    double real_memory_total_mb; // 真实总内存（MB）
    double real_memory_free_mb;  // 真实空闲内存（MB）
    double real_cpu_count;       // 真实CPU核心数
} SystemStatus;

// ============================================
// 高级功能类型定义（用于示例程序）
// ============================================

// 编程语言枚举（示例程序中使用）
typedef enum {
    LANGUAGE_C = 0,             // C语言
    LANGUAGE_CPP = 1,           // C++语言
    LANGUAGE_PYTHON = 2,        // Python语言
    LANGUAGE_JAVA = 3           // Java语言
} Language;

// 代码分析结果（示例程序中使用）
typedef struct {
    int complexity_score;       // 复杂度评分
    double analysis_time;       // 分析时间（秒）
    size_t issue_count;         // 问题数量
    char** issues;              // 问题描述数组
    size_t suggestion_count;    // 建议数量
    char** suggestions;         // 建议描述数组
} CodeAnalysis;

// 以下高级功能类型定义已移至各子模块头文件中：
// - ProductType / ProductRequirement / ProductSpec: product_design/product_design.h
// - DeviceType / DeviceInfo: multisystem/multisystem_control.h
// 示例程序应直接使用上述子模块中的类型定义。

// 能耗数据点（示例程序中使用）
typedef struct {
    double timestamp;           // 时间戳
    double power_watt;          // 功率（瓦特）
    double energy_joule;        // 能量（焦耳）
    double temperature_c;       // 温度（摄氏度）
} EnergyDataPoint;

// 主系统API函数

/**
 * @brief 初始化SELF-LNN系统
 * @param config 系统配置
 * @return 成功返回0，失败返回错误码
 */
SELFLNN_API int selflnn_init(const SystemConfig* config);

/**
 * @brief 关闭系统并释放所有资源
 * @return 成功返回0
 */
SELFLNN_API int selflnn_shutdown(void);

/**
 * @brief 处理多模态输入
 * @param input 输入数据
 * @param state 系统状态（输出）
 * @return 成功返回0，失败返回错误码
 */
SELFLNN_API int selflnn_process_input(const MultimodalInput* input, SystemState* state);

/**
 * @brief 获取系统版本信息
 * @param version 版本信息（输出）
 * @return 成功返回0
 */
SELFLNN_API int selflnn_get_version(VersionInfo* version);

/**
 * @brief 获取当前系统状态
 * @param status 系统状态（输出）
 * @return 成功返回0
 */
SELFLNN_API int selflnn_get_status(SystemStatus* status);

/**
 * @brief 获取最后发生的错误
 * @return 错误码
 */
SELFLNN_API int selflnn_get_last_error(void);

/**
 * @brief 获取错误描述信息
 * @param error_code 错误码
 * @return 错误描述字符串
 */
SELFLNN_API const char* selflnn_get_error_message(int error_code);

/* ================================================================
 * K-030: 配置文件加载与保存
 * ================================================================ */

/**
 * @brief K-030: 从JSON配置文件加载系统配置
 * @param filepath 配置文件路径(NULL使用默认路径)
 * @param config 输出配置
 * @return 0成功，-1失败(使用默认值)
 */
SELFLNN_API int selflnn_config_load_from_file(const char* filepath, SystemConfig* config);

/**
 * @brief K-030: 保存系统配置到JSON文件
 * @param filepath 配置文件路径(NULL使用默认路径)
 * @param config 配置
 * @return 0成功，-1失败
 */
SELFLNN_API int selflnn_config_save_to_file(const char* filepath, const SystemConfig* config);

// ============================================
// 子系统访问器API（供main.c后台任务使用）
// ============================================

/**
 * @brief 获取在线学习器实例句柄（供后台任务使用）
 * @return 在线学习器句柄，未初始化返回NULL
 */
SELFLNN_API void* selflnn_get_online_learner(void);

/**
 * @brief 获取自我演化引擎实例句柄（供后台任务使用）
 * @return 演化引擎句柄，未初始化返回NULL
 */
SELFLNN_API void* selflnn_get_evolution_engine(void);

/**
 * @brief 获取线程池实例句柄
 * @return 线程池句柄，未初始化返回NULL
 */
SELFLNN_API void* selflnn_get_thread_pool(void);

/**
 * @brief 获取自我认知系统实例句柄
 * @return 自我认知系统句柄，未初始化返回NULL
 */
SELFLNN_API void* selflnn_get_self_cognition(void);

/**
 * @brief 获取元认知系统实例句柄
 * @return 元认知系统句柄，未初始化返回NULL
 */
SELFLNN_API void* selflnn_get_metacognition(void);

/**
 * @brief 获取对话处理器实例句柄
 * @return 对话处理器句柄，未初始化返回NULL
 */
SELFLNN_API void* selflnn_get_dialogue_processor(void);

/**
 * @brief 获取安全监控系统实例句柄
 * @return 安全监控句柄，未初始化返回NULL
 */
SELFLNN_API void* selflnn_get_safety_monitor(void);

/**
 * @brief 获取全局唯一液态神经网络实例（单一模型原则）
 * 
 * 整个系统只允许存在一个LNN实例，所有模块通过此函数获取共享引用。
 * 任何模块禁止自行调用 lnn_create() 创建独立LNN。
 *
 * @return LNN句柄，未初始化返回NULL
 */
SELFLNN_API void* selflnn_get_lnn(void);

/**
 * @brief 锁定单一LNN模式 —— 禁止任何子系统独立创建LNN
 * 
 * F-017: 调用后 g_single_lnn_enforced = 1
 * 所有后续的 lnn_create() 调用将被重定向到全局LNN。
 */
SELFLNN_API void selflnn_enforce_single_lnn(void);

/**
 * @brief 检查是否已启用单一LNN强制执行
 * 
 * @return 1=已强制, 0=未强制
 */
SELFLNN_API int selflnn_is_single_lnn_enforced(void);

/**
 * @brief 获取或创建全局唯一LNN（单一模型原则）
 * 
 * 如果已存在全局LNN则返回它，否则用给定配置创建新的全局LNN。
 * 这是创建LNN的首选方式（而非直接调用 lnn_create()）。
 * 
 * @param config LNN配置（仅在首次创建时使用）
 * @return LNN句柄，失败返回NULL
 */
SELFLNN_API LNN* selflnn_get_or_create_lnn(const LNNConfig* config);

/**
 * @brief 获取全局统一LNN状态（多模态共享同一连续动态系统）
 *
 * 所有模态通过统一LNN状态共享同一CfC连续动态系统。
 * 任何模块禁止自行创建独立的CfC细胞或LNN实例处理多模态数据。
 *
 * @return UnifiedLNNState句柄，未初始化返回NULL
 */
SELFLNN_API void* selflnn_get_unified_lnn_state(void);

/**
 * @brief 获取全局共享LNN（单一模型原则，别名）
 * @return LNN句柄
 */
SELFLNN_API void* selflnn_get_shared_lnn(void);

/**
 * @brief 获取统一多模态状态（单一模型原则，别名）
 * @return UnifiedLNNState句柄
 */
SELFLNN_API void* selflnn_get_unified_state(void);

/**
 * @brief 注册子系统模块到全局注册表（单一LNN原则）
 * @param module_id 模块ID (0-20)
 * @param instance 模块实例指针
 * @param uses_shared_lnn 是否使用全局共享LNN（必须为1，除非是无LNN的纯管理模块）
 * @return 成功返回0
 */
SELFLNN_API int selflnn_register_module(int module_id, void* instance, int uses_shared_lnn);

/**
 * @brief 检查模块是否使用共享LNN
 * @param module_id 模块ID
 * @return 1=使用共享LNN, 0=独立
 */
SELFLNN_API int selflnn_module_uses_shared_lnn(int module_id);

SELFLNN_API void* selflnn_get_planning_system(void);

SELFLNN_API void* selflnn_get_auto_learning(void);

SELFLNN_API void* selflnn_get_data_pipeline(void);

SELFLNN_API void* selflnn_get_speech_recognizer(void);

SELFLNN_API void selflnn_set_speech_recognizer(void* sr);

// ============================================
// 高级功能API（用于示例程序）
// ============================================

/**
 * @brief 分析源代码
 * @param source_code 源代码字符串
 * @param language 编程语言类型
 * @param analysis 分析结果输出
 * @return 成功返回0，失败返回错误码
 */
SELFLNN_API int selflnn_analyze_code(const char* source_code, Language language, CodeAnalysis* analysis);

/**
 * @brief 设计产品
 * @param requirement 产品需求
 * @param design 产品设计输出
 * @return 成功返回0，失败返回错误码
 */
SELFLNN_API int selflnn_design_product(const ProductRequirement* requirement, ProductSpec* design);

/**
 * @brief 发现设备
 * @param devices 设备信息数组（输出）
 * @param max_devices 数组最大容量
 * @return 发现的设备数量
 */
SELFLNN_API size_t selflnn_discover_devices(DeviceInfo* devices, size_t max_devices);

/**
 * @brief 设置功率模式
 * @param power_mode 功率模式
 * @return 成功返回0，失败返回错误码
 */
SELFLNN_API int selflnn_set_power_mode(PowerMode power_mode);

/**
 * @brief 监控能耗
 * @param duration 监控时长（秒）
 * @param data_points 数据点数组（输出）
 * @param max_points 数组最大容量
 * @return 采集到的数据点数量
 */
SELFLNN_API size_t selflnn_monitor_energy(double duration, EnergyDataPoint* data_points, size_t max_points);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_H