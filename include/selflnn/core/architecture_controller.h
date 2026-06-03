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
    ARCH_CHANGE_NONE = 0,              /**< 无变更（占位） */
    ARCH_CHANGE_EXPAND_HIDDEN,         /**< 扩展隐藏层宽度（增加神经元） */
    ARCH_CHANGE_SHRINK_HIDDEN,         /**< 收缩隐藏层宽度（移除低激活神经元） */
    ARCH_CHANGE_ADD_LAYER,             /**< 增加网络层（在末尾追加新层） */
    ARCH_CHANGE_REMOVE_LAYER,          /**< 删除网络层（移除最后层） */
    ARCH_CHANGE_RESHAPE_ALL,           /**< 完全重建（改变所有维度参数） */
    ARCH_CHANGE_REPLACE_ARCHITECTURE,  /**< 用外部架构描述替换（NAS部署用） */
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

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_ARCHITECTURE_CONTROLLER_H */
