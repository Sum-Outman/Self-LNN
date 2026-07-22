/**
 * @file architecture_controller.h
 * @brief 动态架构控制器 —— 运行时安全地修改液态神经网络结构
 *
 * 核心解决的问题（架构动态演化能力深度审计报告 P0 缺陷）：
 *   1. 网络结构完全固化 —— 现在可以在运行时增加/删除神经元和层
 *   2. 自我修正架构路径是死代码 —— 提供结构化的架构变更提交通道
 *   3. NAS 搜索结果从不部署 —— 提供 nas_deploy_architecture 桥接函数
 *   4. 演化引擎无结构变异 —— 提供结构变异操作的中转接口
 *
 * 安全机制：
 *   - 所有架构变更必须经过安全审批（arch_controller_approve_change）
 *   - 变更前自动保存检查点（防止不可逆破坏）
 *   - 知识迁移确保权重在新架构中保留最大信息
 *   - 原子交换（old→new）保证推理不中断
 *   - 变更历史完整审计日志
 */

#ifndef SELFLNN_ARCHITECTURE_CONTROLLER_H
#define SELFLNN_ARCHITECTURE_CONTROLLER_H

#include <stddef.h>
#include <stdint.h>
#include "selflnn/core/common.h"
#include "selflnn/core/lnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============ 架构变更类型枚举 ============ */

/**
 * @brief 架构变更类型
 *
 * 每种类型对应一种具体的网络结构调整操作。
 * 变更可组合：例如 EXPAND_HIDDEN + ADD_LAYER 可同时执行。
 */
typedef enum {
    ARCH_CHANGE_NONE = 0,              /**< 无变更（默认值/无操作） */
    ARCH_CHANGE_EXPAND_HIDDEN,         /**< 扩展隐藏层宽度（增加神经元） */
    ARCH_CHANGE_SHRINK_HIDDEN,         /**< 收缩隐藏层宽度（移除低激活神经元） */
    ARCH_CHANGE_ADD_LAYER,             /**< 增加网络层（在末尾追加新层） */
    ARCH_CHANGE_REMOVE_LAYER,          /**< 删除网络层（移除最后层） */
    ARCH_CHANGE_RESHAPE_ALL,           /**< 完全重建（改变所有维度参数） */
    ARCH_CHANGE_REPLACE_ARCHITECTURE,  /**< 用外部架构描述替换（NAS部署用） */
    ARCH_CHANGE_ROLLBACK,              /**< 安全回滚操作（D-001修复：P0编译错误修复） */
} ArchitectureChangeType;

/* ============ 架构变更请求与结果 ============ */

/**
 * @brief 架构变更请求
 *
 * 由子系统（自我认知/NAS/API/演化引擎）提交给架构控制器。
 * 控制器负责审批、执行和记录。
 */
typedef struct {
    ArchitectureChangeType type;       /**< 变更类型 */
    size_t  target_hidden_size;        /**< 目标隐藏层大小（0=不变） */
    int     target_num_layers;         /**< 目标层数（0=不变） */
    size_t  target_input_size;         /**< 目标输入维度（0=不变） */
    size_t  target_output_size;        /**< 目标输出维度（0=不变） */
    float   confidence;                /**< 置信度 [0,1]，低于阈值(默认0.6)的请求被拒绝 */
    float   expected_improvement;      /**< 预期性能提升比例 */
    char    reason[256];               /**< 变更原因（用于审计日志） */
    char    source_module[64];         /**< 发起变更的模块名称 */
    void*   external_arch_data;        /**< 外部架构描述数据（ARCH_CHANGE_REPLACE_ARCHITECTURE时用） */
    size_t  external_arch_data_size;   /**< 外部架构数据大小 */
} ArchitectureChangeRequest;

/**
 * @brief 架构变更结果
 */
typedef struct {
    int     success;                   /**< 0=成功，非0=失败 */
    int     error_code;                /**< 失败时的错误码 */
    char    error_message[256];        /**< 失败时的错误描述 */
    char    archive_path[512];         /**< 变更前自动保存的检查点路径 */
    float   actual_improvement;        /**< 变更后首次评估的性能变化 */
    size_t  old_neuron_count;          /**< 变更前神经元总数 */
    size_t  new_neuron_count;          /**< 变更后神经元总数 */
    size_t  old_param_count;           /**< 变更前可学习参数总数 */
    size_t  new_param_count;           /**< 变更后可学习参数总数 */
    uint64_t change_timestamp;         /**< 变更时间戳 */
} ArchitectureChangeResult;

/**
 * @brief 架构控制器配置
 */
typedef struct {
    float   min_confidence_threshold;  /**< 最低信任阈值，默认 0.6 */
    int     max_changes_per_hour;      /**< 每小时最大变更次数，默认 3 */
    int     enable_auto_approval;      /**< 是否启用自动审批（低置信度请求仍需手动确认） */
    int     enable_archive_backup;     /**< 变更前是否自动保存检查点，默认 1 */
    int     enable_knowledge_transfer; /**< 是否启用权重知识迁移，默认 1 */
    char    archive_dir[512];          /**< 检查点存档目录 */
} ArchitectureControllerConfig;

/**
 * @brief 架构变更历史记录
 */
typedef struct {
    ArchitectureChangeRequest  request;       /**< 变更请求 */
    ArchitectureChangeResult   result;        /**< 变更结果 */
    char    timestamp_str[32];                /**< 可读时间戳 */
} ArchitectureChangeHistoryEntry;

/* ============ 架构控制器不透明句柄 ============ */

typedef struct ArchitectureController ArchitectureController;

/* ============ 生命周期 ============ */

/**
 * @brief 创建架构控制器实例
 *
 * @param config 控制器配置（NULL=使用默认配置）
 * @return 控制器句柄，失败返回 NULL
 */
ArchitectureController* arch_controller_create(const ArchitectureControllerConfig* config);

/**
 * @brief 销毁架构控制器实例
 *
 * @param controller 控制器句柄
 */
void arch_controller_free(ArchitectureController* controller);

/* ============ 核心变更操作 ============ */

/**
 * @brief 提交架构变更请求
 *
 * 完整的变更流程：
 *   1. 验证请求有效性（维度范围检查）
 *   2. 安全审批（频率限制、置信度检查、安全系统审核）
 *   3. 自动存档（保存当前网络检查点）
 *   4. 创建新 LNN 实例（按目标配置）
 *   5. 知识迁移（将旧网络权重映射到新网络）
 *   6. 原子交换（替换 g_system_state 中的 LNN 实例）
 *   7. 销毁旧 LNN
 *   8. 记录变更历史
 *
 * @param controller 控制器句柄
 * @param lnn 当前 LNN 实例指针的指针（原子交换后会被替换）
 * @param request 变更请求
 * @param result 变更结果输出（可为 NULL）
 * @return SELFLNN_SUCCESS(0)=成功，其它=失败
 */
int arch_controller_submit_change(ArchitectureController* controller,
                                   LNN** lnn,
                                   const ArchitectureChangeRequest* request,
                                   ArchitectureChangeResult* result);

/**
 * @brief 原子重建 LNN 为新架构（保留知识迁移）
 *
 * 不经过审批流程，直接按新配置重建网络。
 * 仅用于系统初始化或可信的内部调用。
 *
 * @param old_lnn 旧 LNN 实例
 * @param new_config 新配置
 * @return 新 LNN 实例，失败返回 NULL（旧实例保持不变）
 */
LNN* arch_controller_rebuild_lnn(LNN* old_lnn, const LNNConfig* new_config);

/**
 * @brief 知识迁移：将旧网络的权重映射到新网络
 *
 * 迁移策略按变更类型：
 *   - 扩展隐藏层：旧权重复制到新网络左上角，新增部分 Xavier 初始化
 *   - 收缩隐藏层：取旧网络权重左上角子矩阵
 *   - 增加层：复制已有层，新层随机初始化
 *   - 删除层：跳过最后一层
 *   - 完全重建：保留输出投影层权重（尽可能保留语义空间）
 *
 * @param old_lnn 旧 LNN 实例
 * @param new_lnn 新 LNN 实例（已按新配置创建）
 * @param change_type 变更类型（用于选择迁移策略）
 * @return 0=成功，-1=失败
 */
int arch_controller_transfer_knowledge(LNN* old_lnn, LNN* new_lnn,
                                        ArchitectureChangeType change_type);

/* ============ 安全审批 ============ */

/**
 * @brief 架构变更审批
 *
 * 审批流程：
 *   1. 频率限制检查（每小时最大变更次数）
 *   2. 置信度检查（低于 threshold 拒绝）
 *   3. 维度合法性检查（hidden_size 等是否在合理范围内）
 *   4. 变更类型合法性检查
 *   5. 安全检查（可调用安全系统回调）
 *
 * @param controller 控制器句柄
 * @param request 变更请求
 * @return 0=审批通过，非0=拒绝
 */
int arch_controller_approve_change(ArchitectureController* controller,
                                    const ArchitectureChangeRequest* request);

/* ============ 变更历史与审计 ============ */

/**
 * @brief 获取变更历史记录数
 *
 * @param controller 控制器句柄
 * @return 历史记录数量
 */
size_t arch_controller_get_history_count(const ArchitectureController* controller);

/**
 * @brief 查询指定索引的变更历史
 *
 * @param controller 控制器句柄
 * @param index 索引（0=最近一次）
 * @param entry 历史记录输出
 * @return 0=成功，-1=索引无效
 */
int arch_controller_get_history_entry(const ArchitectureController* controller,
                                       size_t index,
                                       ArchitectureChangeHistoryEntry* entry);

/* ============ 状态查询 ============ */

/**
 * @brief 获取当前网络架构统计信息（神经元数+参数数）
 *
 * @param lnn LNN 实例
 * @param neuron_count 神经元总数输出
 * @param param_count 可学习参数总数输出
 * @param hidden_size 当前隐藏层大小输出
 * @param num_layers 当前层数输出
 * @return 0=成功，-1=失败
 */
int arch_controller_get_architecture_stats(const LNN* lnn,
                                            size_t* neuron_count,
                                            size_t* param_count,
                                            size_t* hidden_size,
                                            int* num_layers);

/**
 * @brief 获取默认控制器配置
 *
 * @return 填充了默认值的配置
 */
ArchitectureControllerConfig arch_controller_default_config(void);

/**
 * @brief 获取默认架构变更请求
 *
 * @return 填充了默认值的请求（type=NONE）
 */
ArchitectureChangeRequest arch_controller_default_request(void);

/**
 * @brief P0-002修复: 回滚最近一次架构变更
 *
 * 从存档路径加载变更前保存的检查点文件，恢复到变更前的LNN状态。
 * 如果变更后新LNN出现问题，调用此函数可安全回退。
 *
 * @param controller 控制器句柄
 * @param lnn_ptr LNN实例指针的指针（将被替换为回滚后的LNN）
 * @param archive_path 变更前保存的检查点路径
 * @return 0=回滚成功，-1=失败
 */
int arch_controller_rollback(ArchitectureController* controller,
                              LNN** lnn_ptr,
                              const char* archive_path);

/* ============ P2-001: NAS 架构部署桥接 ============ */

/**
 * @brief 将外部架构描述部署到运行中的 LNN
 *
 * 这是打通 NAS→LNN 断裂点的关键函数。
 * 将 NAS 搜索出的 ArchitectureDescription 转换为 ArchitectureChangeRequest
 * 并提交给控制器执行。
 *
 * @param controller 架构控制器句柄
 * @param lnn 当前 LNN 实例指针的指针
 * @param arch_data 架构描述数据（NAS 搜索结果的序列化表示）
 * @param arch_data_size 数据大小
 * @param confidence 部署置信度
 * @param result 部署结果输出（可为 NULL）
 * @return SELFLNN_SUCCESS(0)=成功
 */
int arch_controller_deploy_architecture(ArchitectureController* controller,
                                         LNN** lnn,
                                         const void* arch_data,
                                         size_t arch_data_size,
                                         float confidence,
                                         ArchitectureChangeResult* result);

/* ============ P2-002: 结构变异中转接口 ============ */

/**
 * @brief 结构变异请求类型（演化引擎用）
 */
typedef enum {
    STRUCT_MUTATE_ADD_NEURON,          /**< 随机位置增加一定比例神经元 */
    STRUCT_MUTATE_REMOVE_NEURON,       /**< 移除激活度最低的一定比例神经元 */
    STRUCT_MUTATE_ADD_LAYER,           /**< 在末尾追加新的隐藏层 */
    STRUCT_MUTATE_REMOVE_LAYER,        /**< 移除最后一层 */
    STRUCT_MUTATE_EXPAND_RATIO,        /**< 按比例扩展所有隐藏层宽度 */
    STRUCT_MUTATE_SHRINK_RATIO,        /**< 按比例收缩所有隐藏层宽度 */
} StructuralMutationType;

/**
 * @brief 结构变异参数
 */
typedef struct {
    StructuralMutationType mut_type;   /**< 变异类型 */
    float   mutation_ratio;            /**< 变异比例（如 0.25 = 扩展25%） */
    float   mutation_probability;      /**< 每次评估时执行该变异的概率 [0,1] */
    int     min_hidden_size;           /**< 隐藏层最小尺寸限制 */
    int     max_hidden_size;           /**< 隐藏层最大尺寸限制 */
    int     min_layers;                /**< 最小层数限制 */
    int     max_layers;                /**< 最大层数限制 */
} StructuralMutationConfig;

/**
 * @brief 执行结构变异（演化引擎回调接口）
 *
 * 根据变异参数生成 ArchitectureChangeRequest 并提交。
 * 返回的 result 中包含变更前后的神经元/参数统计。
 *
 * @param controller 控制器句柄
 * @param lnn 当前 LNN 实例指针的指针
 * @param config 结构变异配置
 * @param result 变异结果输出（可为 NULL）
 * @return SELFLNN_SUCCESS(0)=成功
 */
int arch_controller_structural_mutate(ArchitectureController* controller,
                                       LNN** lnn,
                                       const StructuralMutationConfig* config,
                                       ArchitectureChangeResult* result);

/* ================================================================
 * H-004修复: 运行时激活度监控与在线结构调整
 * 实现需求R49: "当任务复杂度升高时自动增加神经元、
 * 移除冗余或低激活度神经元、调整层数"
 *
 * 核心设计:
 *   1. 激活度监控器(ArchActivationMonitor) - 在推理过程中持续收集
 *      每层神经元的激活统计(EMA均值/方差/最小/最大)
 *   2. 任务复杂度评估器 - 基于激活熵、激活幅度方差、梯度范数
 *      综合评估当前任务复杂度
 *   3. 在线触发机制 - 每N次推理后自动评估是否需要结构调整
 *   4. 低激活度检测 - 识别持续低于阈值的神经元用于修剪
 *   5. 饱和检测 - 识别持续饱和的神经元触发扩展
 * ================================================================ */

/** @brief 单层激活统计 */
typedef struct {
    float*  ema_mean;          /**< 指数移动平均均值 [hidden_size] */
    float*  ema_variance;      /**< 指数移动平均方差 [hidden_size] */
    float*  ema_min;           /**< 指数移动平均最小值 [hidden_size] */
    float*  ema_max;           /**< 指数移动平均最大值 [hidden_size] */
    size_t  sample_count;      /**< 采样计数 */
    size_t  hidden_size;       /**< 该层隐藏维度 */
    float   ema_decay;         /**< EMA衰减因子 (默认0.99) */
} ArchLayerActivationStats;

/** @brief 激活度监控器 */
typedef struct {
    ArchLayerActivationStats* layers;     /**< 每层激活统计 [num_layers] */
    int     num_layers;                   /**< 监控的层数 */
    size_t  total_samples;                /**< 总采样数 */
    int     check_interval;               /**< 每隔多少次推理检查一次 (默认100) */
    float   low_activity_threshold;       /**< 低激活度阈值 (默认0.01) */
    float   saturation_threshold;         /**< 饱和阈值 (默认0.95) */
    float   low_activity_ratio_trigger;   /**< 低激活神经元比例触发修剪 (默认0.15) */
    float   saturation_ratio_trigger;     /**< 饱和神经元比例触发扩展 (默认0.10) */
    int     consecutive_checks_required;  /**< 连续检查次数要求 (默认3) */
    int     is_initialized;               /**< 初始化标志 */
    float   current_complexity_score;     /**< 当前任务复杂度评分 */
    float   complexity_history[16];       /**< 复杂度历史环形缓冲 */
    int     complexity_history_idx;       /**< 环形缓冲索引 */
    int     complexity_history_count;     /**< 有效历史条目数 */
} ArchActivationMonitor;

/** @brief 任务复杂度指标 */
typedef struct {
    float   activation_entropy;           /**< 激活熵 (越高=越复杂) */
    float   activation_variance;          /**< 激活幅度方差 */
    float   gradient_norm;                /**< 最近梯度范数 (0=不可用) */
    float   saturation_ratio;             /**< 饱和神经元比例 */
    float   low_activity_ratio;           /**< 低激活度神经元比例 */
    float   overall_complexity;           /**< 综合复杂度评分 [0,1] */
    int     needs_expansion;              /**< 是否需要扩展神经元 */
    int     needs_pruning;                /**< 是否需要修剪神经元 */
    int     needs_layer_adjustment;       /**< 是否需要调整层数 */
    char    diagnosis[256];               /**< 诊断说明 */
} ArchComplexityMetrics;

/** @brief 在线调整建议 */
typedef struct {
    int     recommend_expand;             /**< 建议扩展神经元 */
    int     recommend_prune;              /**< 建议修剪神经元 */
    int     recommend_add_layer;          /**< 建议增加层 */
    int     recommend_remove_layer;       /**< 建议移除层 */
    size_t  target_hidden_size;           /**< 建议目标隐藏层大小 */
    int     target_num_layers;            /**< 建议目标层数 */
    float   confidence;                   /**< 建议置信度 */
    char    reason[256];                  /**< 建议原因 */
    size_t  low_activity_count;           /**< 低激活度神经元数量 */
    size_t  saturated_count;              /**< 饱和神经元数量 */
} ArchOnlineAdaptationAdvice;

/**
 * @brief 创建激活度监控器
 *
 * @param num_layers 监控的层数
 * @param hidden_sizes 每层隐藏维度数组 [num_layers]
 * @param check_interval 每隔多少次推理检查一次
 * @return 监控器实例，失败返回NULL
 */
ArchActivationMonitor* arch_controller_create_activation_monitor(
    int num_layers,
    const size_t* hidden_sizes,
    int check_interval);

/**
 * @brief 销毁激活度监控器
 */
void arch_controller_free_activation_monitor(ArchActivationMonitor* monitor);

/**
 * @brief 记录一次推理的激活值
 *
 * 在每次lnn_forward后调用，传入每层的激活向量。
 * 使用EMA更新统计信息。
 *
 * @param monitor 监控器
 * @param layer_idx 层索引
 * @param activations 激活值数组 [hidden_size]
 * @param hidden_size 该层隐藏维度
 */
void arch_controller_record_activation(ArchActivationMonitor* monitor,
                                        int layer_idx,
                                        const float* activations,
                                        size_t hidden_size);

/**
 * @brief 评估当前任务复杂度
 *
 * 综合分析激活熵、方差、饱和度和低激活度比例，
 * 给出综合复杂度评分和诊断建议。
 *
 * @param monitor 监控器
 * @param metrics 复杂度指标输出
 * @return 0=成功，-1=失败
 */
int arch_controller_assess_complexity(ArchActivationMonitor* monitor,
                                       ArchComplexityMetrics* metrics);

/**
 * @brief 检测低激活度神经元
 *
 * 扫描所有层，找出持续低于阈值的神经元索引。
 *
 * @param monitor 监控器
 * @param low_activity_count 低激活度神经元总数输出
 * @param saturated_count 饱和神经元总数输出
 * @return 0=成功
 */
int arch_controller_detect_activity_anomalies(ArchActivationMonitor* monitor,
                                               size_t* low_activity_count,
                                               size_t* saturated_count);

/**
 * @brief 在线自适应调整评估
 *
 * 综合激活监控数据，评估是否需要在线调整架构。
 * 不直接执行变更，而是返回建议供调用者决策。
 *
 * @param monitor 监控器
 * @param lnn 当前LNN实例
 * @param advice 调整建议输出
 * @return 0=成功，-1=失败
 */
int arch_controller_evaluate_online_adaptation(ArchActivationMonitor* monitor,
                                                const LNN* lnn,
                                                ArchOnlineAdaptationAdvice* advice);

/**
 * @brief 执行在线自适应调整
 *
 * 结合监控数据和建议，自动执行神经元增长/修剪/层调整。
 * 完整的在线调整流程:
 *   1. 评估当前激活状态
 *   2. 生成调整建议
 *   3. 如果置信度足够，提交架构变更
 *   4. 重置激活统计以开始新一轮监控
 *
 * @param controller 架构控制器
 * @param lnn_ptr LNN实例指针的指针
 * @param monitor 激活监控器
 * @param result 调整结果输出
 * @return 0=成功/无需调整，-1=失败
 */
int arch_controller_online_adapt(ArchitectureController* controller,
                                  LNN** lnn_ptr,
                                  ArchActivationMonitor* monitor,
                                  ArchitectureChangeResult* result);

/**
 * @brief 重置激活统计
 *
 * 在架构变更后调用，清除旧网络的激活统计。
 */
void arch_controller_reset_activation_stats(ArchActivationMonitor* monitor);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_ARCHITECTURE_CONTROLLER_H */
