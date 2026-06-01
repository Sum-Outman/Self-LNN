/**
 * @file multisystem_control.h
 * @brief 多系统控制能力接口
 * 
 * 多系统控制能力核心接口，支持设备发现、任务分配、协调控制和状态监控。
 */

#ifndef SELFLNN_MULTISYSTEM_CONTROL_H
#define SELFLNN_MULTISYSTEM_CONTROL_H

#include <stddef.h>
#include "selflnn/multisystem/swarm_intelligence.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 设备类型枚举
 */
typedef enum {
    DEVICE_TYPE_ROBOT = 0,       /**< 机器人设备 */
    DEVICE_TYPE_SENSOR = 1,      /**< 传感器设备 */
    DEVICE_TYPE_ACTUATOR = 2,    /**< 执行器设备 */
    DEVICE_TYPE_COMPUTE = 3,     /**< 计算设备 */
    DEVICE_TYPE_STORAGE = 4,     /**< 存储设备 */
    DEVICE_TYPE_NETWORK = 5,     /**< 网络设备 */
    DEVICE_TYPE_CUSTOM = 6       /**< 自定义设备 */
} DeviceType;

/**
 * @brief 设备状态枚举
 */
typedef enum {
    DEVICE_STATE_IDLE = 0,       /**< 空闲状态 */
    DEVICE_STATE_BUSY = 1,       /**< 忙碌状态 */
    DEVICE_STATE_ERROR = 2,      /**< 错误状态 */
    DEVICE_STATE_OFFLINE = 3,    /**< 离线状态 */
    DEVICE_STATE_MAINTENANCE = 4 /**< 维护状态 */
} DeviceState;

/**
 * @brief 设备信息结构体
 */
typedef struct {
    char* device_id;             /**< 设备唯一标识 */
    DeviceType type;             /**< 设备类型 */
    DeviceState state;           /**< 设备状态 */
    char* name;                  /**< 设备名称 */
    char* description;           /**< 设备描述 */
    char* model;                 /**< 设备型号 */
    double last_seen;            /**< 最后检测时间（秒） */
    int is_online;               /**< 是否在线（0=离线，1=在线） */
    double capability_score;     /**< 能力评分（0-1） */
    double reliability_score;    /**< 可靠性评分（0-1） */
    double current_load;         /**< 当前负载（0-1） */
    void* device_specific_data;  /**< 设备特定数据 */
} DeviceInfo;

/**
 * @brief 任务类型枚举
 */
typedef enum {
    TASK_TYPE_COMPUTE = 0,       /**< 计算任务 */
    TASK_TYPE_MOVE = 1,          /**< 移动任务 */
    TASK_TYPE_SENSE = 2,         /**< 感知任务 */
    TASK_TYPE_ACTUATE = 3,       /**< 执行任务 */
    TASK_TYPE_COMMUNICATE = 4,   /**< 通信任务 */
    TASK_TYPE_MONITOR = 5,       /**< 监控任务 */
    TASK_TYPE_CUSTOM = 6         /**< 自定义任务 */
} TaskType;

/**
 * @brief 任务结构体
 */
typedef struct {
    char* task_id;               /**< 任务唯一标识 */
    TaskType type;               /**< 任务类型 */
    char* description;           /**< 任务描述 */
    double priority;             /**< 任务优先级（0-1） */
    double estimated_duration;   /**< 预估耗时（秒） */
    double required_capability;  /**< 所需能力评分（0-1） */
    DeviceType preferred_device_type; /**< 偏好设备类型 */
    void* task_data;             /**< 任务数据 */
    char name[256];              /**< 任务名称（AGI执行委派用） */
    float params[1024];          /**< 任务参数缓冲区（AGI执行委派用） */
    int param_count;             /**< 任务参数数量 */
    double deadline;             /**< 任务截止时间 */
} Task;

/**
 * @brief 任务分配结果
 */
typedef struct {
    char* task_id;               /**< 任务ID */
    char* device_id;             /**< 分配的设备ID */
    double assignment_score;     /**< 分配评分（0-1） */
    double estimated_completion_time; /**< 预估完成时间 */
    char** alternative_devices;  /**< 备选设备列表 */
    size_t alternative_count;    /**< 备选设备数量 */
} TaskAssignment;

/**
 * @brief 系统协调计划
 */
typedef struct {
    TaskAssignment* assignments; /**< 任务分配数组 */
    size_t assignment_count;     /**< 分配数量 */
    double total_estimated_time; /**< 总预估时间 */
    double system_efficiency;    /**< 系统效率评分（0-1） */
    double load_balance_score;   /**< 负载均衡评分（0-1） */
} CoordinationPlan;

/**
 * @brief 多系统控制引擎句柄
 */
typedef struct MultiSystemControlEngine MultiSystemControlEngine;

/**
 * @brief 创建多系统控制引擎
 * 
 * @return MultiSystemControlEngine* 引擎句柄，失败返回NULL
 */
MultiSystemControlEngine* multisystem_control_engine_create(void);

/**
 * @brief 销毁多系统控制引擎
 * 
 * @param engine 引擎句柄
 */
void multisystem_control_engine_destroy(MultiSystemControlEngine* engine);

/**
 * @brief 发现可用设备
 * 
 * @param engine 引擎句柄
 * @param device_list 设备列表输出（需要调用者释放）
 * @param device_count 设备数量输出
 * @return int 成功返回0，失败返回错误码
 */
int discover_available_devices(MultiSystemControlEngine* engine,
                               DeviceInfo*** device_list,
                               size_t* device_count);

/**
 * @brief 释放设备列表
 * 
 * @param device_list 设备列表
 * @param device_count 设备数量
 */
void free_device_list(DeviceInfo** device_list, size_t device_count);

/**
 * @brief 创建设备信息
 * 
 * @param device_id 设备ID
 * @param type 设备类型
 * @param name 设备名称
 * @return DeviceInfo* 设备信息，失败返回NULL
 */
DeviceInfo* create_device_info(const char* device_id, DeviceType type, const char* name);

/**
 * @brief 销毁设备信息
 * 
 * @param device_info 设备信息
 */
void destroy_device_info(DeviceInfo* device_info);

/**
 * @brief 创建任务
 * 
 * @param task_id 任务ID
 * @param type 任务类型
 * @param description 任务描述
 * @return Task* 任务，失败返回NULL
 */
Task* create_task(const char* task_id, TaskType type, const char* description);

/**
 * @brief 销毁任务
 * 
 * @param task 任务
 */
void destroy_task(Task* task);

/**
 * @brief 分配任务到设备
 * 
 * @param engine 引擎句柄
 * @param task 任务
 * @param available_devices 可用设备列表
 * @param device_count 设备数量
 * @return TaskAssignment* 任务分配结果，失败返回NULL
 */
TaskAssignment* assign_task_to_device(MultiSystemControlEngine* engine,
                                      const Task* task,
                                      DeviceInfo** available_devices,
                                      size_t device_count);

/**
 * @brief 销毁任务分配
 * 
 * @param assignment 任务分配
 */
void destroy_task_assignment(TaskAssignment* assignment);

/**
 * @brief 协调多任务执行
 * 
 * @param engine 引擎句柄
 * @param tasks 任务数组
 * @param task_count 任务数量
 * @param available_devices 可用设备列表
 * @param device_count 设备数量
 * @return CoordinationPlan* 协调计划，失败返回NULL
 */
CoordinationPlan* coordinate_multitask_execution(MultiSystemControlEngine* engine,
                                                 Task** tasks,
                                                 size_t task_count,
                                                 DeviceInfo** available_devices,
                                                 size_t device_count);

/**
 * @brief 销毁协调计划
 * 
 * @param plan 协调计划
 */
void destroy_coordination_plan(CoordinationPlan* plan);

/**
 * @brief 执行协调计划
 * 
 * @param engine 引擎句柄
 * @param plan 协调计划
 * @return int 成功返回0，失败返回错误码
 */
int execute_coordination_plan(MultiSystemControlEngine* engine,
                              const CoordinationPlan* plan);

/**
 * @brief 监控系统状态
 * 
 * @param engine 引擎句柄
 * @param device_list 设备列表
 * @param device_count 设备数量
 * @return int 成功返回0，失败返回错误码
 */
int monitor_system_status(MultiSystemControlEngine* engine,
                          DeviceInfo** device_list,
                          size_t device_count);

// ==================== 群体智能集成接口 ====================

/**
 * @brief 使用群体智能优化任务分配
 * 
 * @param engine 引擎句柄
 * @param tasks 任务数组
 * @param task_count 任务数量
 * @param devices 可用设备列表
 * @param device_count 设备数量
 * @return int 成功返回0，失败返回-1
 */
int multisystem_swarm_optimize_assignment(MultiSystemControlEngine* engine,
                                          Task** tasks, size_t task_count,
                                          DeviceInfo** devices, size_t device_count);

/**
 * @brief 执行单步群体智能迭代
 * 
 * @param engine 引擎句柄
 * @return int 成功返回0，失败返回-1
 */
int multisystem_swarm_iterate(MultiSystemControlEngine* engine);

/**
 * @brief 获取群体智能状态
 * 
 * @param engine 引擎句柄
 * @param state 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int multisystem_swarm_get_state(MultiSystemControlEngine* engine, MSSwarmState* state);

/**
 * @brief 获取群体智能最佳解
 * 
 * @param engine 引擎句柄
 * @param position 最佳位置输出缓冲区
 * @param fitness 最佳适应度输出
 * @return int 成功返回0，失败返回-1
 */
int multisystem_swarm_get_best(MultiSystemControlEngine* engine,
                               float* position, float* fitness);

/**
 * @brief 重置群体智能优化器
 * 
 * @param engine 引擎句柄
 * @return int 成功返回0，失败返回-1
 */
int multisystem_swarm_reset(MultiSystemControlEngine* engine);

// ==================== TCP RPC 传输层接口 ====================

/**
 * @brief TCP RPC 传输层句柄
 */
typedef struct TcpRpcTransport TcpRpcTransport;

/**
 * @brief 创建TCP RPC传输层
 * 
 * @param local_port 本地监听端口（0=默认18760）
 * @return TcpRpcTransport* 成功返回指针，失败返回NULL
 */
TcpRpcTransport* rpc_transport_create(int local_port);

/**
 * @brief 启动TCP RPC传输层
 * 
 * @param transport 传输层实例
 * @param node_id 本地节点ID（用于消息中的sender_id）
 * @return int 成功返回0，失败返回-1
 */
int rpc_transport_start(TcpRpcTransport* transport, const char* node_id);

/**
 * @brief 停止TCP RPC传输层
 * 
 * @param transport 传输层实例
 * @return int 成功返回0，失败返回-1
 */
int rpc_transport_stop(TcpRpcTransport* transport);

/**
 * @brief 添加对端节点（启动前配置）
 * 
 * @param transport 传输层实例
 * @param host 主机名或IP地址
 * @param port 端口号（0=默认18760）
 * @return int 成功返回0，失败返回-1
 */
int rpc_transport_add_peer(TcpRpcTransport* transport,
                           const char* host, int port);

/**
 * @brief 获取已连接对端数量
 * 
 * @param transport 传输层实例
 * @return int 已连接对端数量
 */
int rpc_transport_get_peer_count(TcpRpcTransport* transport);

/**
 * @brief 判断传输层是否正在运行
 * 
 * @param transport 传输层实例
 * @return int 运行中返回1，否则返回0
 */
int rpc_transport_is_running(TcpRpcTransport* transport);

// ==================== 冲突检测与避免接口 ====================

/**
 * @brief 冲突类型枚举
 */
typedef enum {
    CONFLICT_TYPE_SPATIAL      = 0,  /**< 空间冲突：两个设备试图占据同一物理空间 */
    CONFLICT_TYPE_RESOURCE     = 1,  /**< 资源冲突：多个任务竞争同一独占资源 */
    CONFLICT_TYPE_DEPENDENCY   = 2,  /**< 依赖冲突：任务之间存在循环依赖 */
    CONFLICT_TYPE_TIMING       = 3,  /**< 时序冲突：任务执行时间窗口重叠导致冲突 */
    CONFLICT_TYPE_COMMUNICATION = 4, /**< 通信冲突：多个设备争用同一通信信道 */
    CONFLICT_TYPE_CUSTOM       = 5   /**< 自定义冲突类型 */
} ConflictType;

/**
 * @brief 冲突严重程度枚举
 */
typedef enum {
    CONFLICT_SEVERITY_LOW      = 0,  /**< 低严重度：可忽略或自动解决 */
    CONFLICT_SEVERITY_MEDIUM   = 1,  /**< 中严重度：需要重新调度 */
    CONFLICT_SEVERITY_HIGH     = 2,  /**< 高严重度：必须立即处理 */
    CONFLICT_SEVERITY_CRITICAL = 3   /**< 致命严重度：可能导致系统损坏 */
} ConflictSeverity;

/**
 * @brief 冲突解决策略枚举
 */
typedef enum {
    CONFLICT_RESOLVE_REASSIGN   = 0, /**< 重新分配：将任务分配给其他设备 */
    CONFLICT_RESOLVE_DELAY      = 1, /**< 延迟执行：推迟其中一个任务 */
    CONFLICT_RESOLVE_AVOID      = 2, /**< 路径规避：改变运动路径避免空间冲突 */
    CONFLICT_RESOLVE_CANCEL     = 3, /**< 取消任务：取消低优先级任务 */
    CONFLICT_RESOLVE_NEGOTIATE  = 4, /**< 协商解决：通过分布式协商达成一致 */
    CONFLICT_RESOLVE_PRIORITY   = 5, /**< 优先级排序：按优先级分配资源 */
    CONFLICT_RESOLVE_MERGE      = 6  /**< 合并执行：将可以合并的任务合并执行 */
} ConflictResolution;

/**
 * @brief 空间边界结构体（用于空间冲突检测的AABB包围盒）
 */
typedef struct {
    double min_x;                /**< X轴最小坐标 */
    double min_y;                /**< Y轴最小坐标 */
    double min_z;                /**< Z轴最小坐标 */
    double max_x;                /**< X轴最大坐标 */
    double max_y;                /**< Y轴最大坐标 */
    double max_z;                /**< Z轴最大坐标 */
    char* device_id;             /**< 关联的设备ID */
    char* task_id;               /**< 关联的任务ID */
    double timestamp;            /**< 时间戳 */
    double duration;             /**< 持续时间（秒），0=瞬时 */
} SpatialBounds;

/**
 * @brief 资源预留结构体
 */
typedef struct {
    char* resource_id;           /**< 资源唯一标识（如"gpu:0"、"robot_arm:1"） */
    char* resource_type;         /**< 资源类型（如"gpu"、"arm"、"channel"） */
    char* device_id;             /**< 持有资源的设备ID */
    int exclusive;               /**< 是否独占资源（1=独占，0=可共享） */
    double capacity_total;       /**< 总容量/带宽 */
    double capacity_used;        /**< 已使用容量 */
    double estimated_release;    /**< 预计释放时间 */
} ResourceReservation;

/**
 * @brief 冲突信息结构体
 */
typedef struct {
    ConflictType type;           /**< 冲突类型 */
    ConflictSeverity severity;   /**< 冲突严重程度 */
    ConflictResolution suggested_resolution; /**< 建议的解决策略 */
    char* description;           /**< 冲突描述 */
    char** involved_devices;     /**< 涉及的设备ID列表 */
    size_t involved_device_count; /**< 涉及设备数量 */
    char** involved_tasks;       /**< 涉及的任务ID列表 */
    size_t involved_task_count;  /**< 涉及任务数量 */
    double conflict_value;       /**< 冲突量化值（0-1） */
    double timestamp;            /**< 冲突发现时间 */
    int auto_resolvable;         /**< 是否可自动解决 */
    void* conflict_data;         /**< 冲突特定数据 */
} ConflictInfo;

/**
 * @brief 冲突检测结果结构体
 */
typedef struct {
    ConflictInfo** conflicts;    /**< 冲突数组 */
    size_t conflict_count;       /**< 冲突数量 */
    size_t conflict_capacity;    /**< 冲突数组容量 */
    double total_conflict_score; /**< 综合冲突评分（0-1） */
    int has_critical_conflict;   /**< 是否存在致命冲突 */
    double detection_time_ms;    /**< 检测耗时（毫秒） */
} ConflictDetectionResult;

/**
 * @brief 冲突检测器句柄
 */
typedef struct ConflictDetector ConflictDetector;

/**
 * @brief 创建冲突检测器
 * 
 * @param engine 多系统控制引擎句柄
 * @return ConflictDetector* 冲突检测器句柄，失败返回NULL
 */
ConflictDetector* conflict_detector_create(MultiSystemControlEngine* engine);

/**
 * @brief 销毁冲突检测器
 * 
 * @param detector 冲突检测器句柄
 */
void conflict_detector_destroy(ConflictDetector* detector);

/**
 * @brief 注册空间边界（用于空间冲突检测）
 * 
 * @param detector 冲突检测器句柄
 * @param bounds 空间边界信息
 * @return int 成功返回0，失败返回-1
 */
int conflict_detector_register_spatial_bounds(ConflictDetector* detector,
                                              const SpatialBounds* bounds);

/**
 * @brief 注册资源预留（用于资源冲突检测）
 * 
 * @param detector 冲突检测器句柄
 * @param reservation 资源预留信息
 * @return int 成功返回0，失败返回-1
 */
int conflict_detector_register_resource(ConflictDetector* detector,
                                        const ResourceReservation* reservation);

/**
 * @brief 注册任务依赖关系（用于依赖冲突检测）
 * 
 * @param detector 冲突检测器句柄
 * @param task_id 任务ID
 * @param depends_on_task_id 依赖的任务ID
 * @return int 成功返回0，失败返回-1
 */
int conflict_detector_register_dependency(ConflictDetector* detector,
                                          const char* task_id,
                                          const char* depends_on_task_id);

/**
 * @brief 检测空间冲突：检查所有注册的空间边界是否存在重叠
 * 
 * @param detector 冲突检测器句柄
 * @param current_time 当前系统时间
 * @return ConflictDetectionResult* 检测结果，失败返回NULL（需要调用者释放）
 */
ConflictDetectionResult* conflict_detector_check_spatial(ConflictDetector* detector,
                                                         double current_time);

/**
 * @brief 检测资源冲突：检查所有注册的资源是否存在争用
 * 
 * @param detector 冲突检测器句柄
 * @return ConflictDetectionResult* 检测结果，失败返回NULL
 */
ConflictDetectionResult* conflict_detector_check_resources(ConflictDetector* detector);

/**
 * @brief 检测依赖冲突：检查任务依赖图中是否存在循环依赖
 * 
 * @param detector 冲突检测器句柄
 * @return ConflictDetectionResult* 检测结果，失败返回NULL
 */
ConflictDetectionResult* conflict_detector_check_dependencies(ConflictDetector* detector);

/**
 * @brief 检测时序冲突：检查任务时间窗口是否存在冲突
 * 
 * @param detector 冲突检测器句柄
 * @param tasks 任务数组
 * @param task_count 任务数量
 * @return ConflictDetectionResult* 检测结果，失败返回NULL
 */
ConflictDetectionResult* conflict_detector_check_timing(ConflictDetector* detector,
                                                        Task** tasks,
                                                        size_t task_count);

/**
 * @brief 运行全部冲突检测（空间+资源+依赖+时序）
 * 
 * @param detector 冲突检测器句柄
 * @param tasks 任务数组
 * @param task_count 任务数量
 * @param current_time 当前系统时间
 * @return ConflictDetectionResult* 综合检测结果，失败返回NULL
 */
ConflictDetectionResult* conflict_detector_run_all(ConflictDetector* detector,
                                                   Task** tasks,
                                                   size_t task_count,
                                                   double current_time);

/**
 * @brief 释放冲突检测结果
 * 
 * @param result 检测结果
 */
void conflict_detector_free_result(ConflictDetectionResult* result);

/**
 * @brief 自动解决冲突
 * 
 * @param detector 冲突检测器句柄
 * @param result 冲突检测结果
 * @param engine 多系统控制引擎（用于重新分配）
 * @return int 成功解决的冲突数量，失败返回-1
 */
int conflict_detector_resolve_all(ConflictDetector* detector,
                                  const ConflictDetectionResult* result,
                                  MultiSystemControlEngine* engine);

/**
 * @brief 解决单个冲突
 * 
 * @param detector 冲突检测器句柄
 * @param conflict 冲突信息
 * @param engine 多系统控制引擎
 * @return int 成功返回0，失败返回-1
 */
int conflict_detector_resolve_one(ConflictDetector* detector,
                                  const ConflictInfo* conflict,
                                  MultiSystemControlEngine* engine);

/**
 * @brief 清除所有注册的空间边界
 * 
 * @param detector 冲突检测器句柄
 */
void conflict_detector_clear_spatial(ConflictDetector* detector);

/**
 * @brief 清除所有注册的资源
 * 
 * @param detector 冲突检测器句柄
 */
void conflict_detector_clear_resources(ConflictDetector* detector);

/**
 * @brief 清除所有注册的依赖关系
 * 
 * @param detector 冲突检测器句柄
 */
void conflict_detector_clear_dependencies(ConflictDetector* detector);

/**
 * @brief 设置冲突检测参数（阈值等）
 * 
 * @param detector 冲突检测器句柄
 * @param spatial_threshold 空间重叠阈值（0-1），默认0.1
 * @param resource_threshold 资源争用阈值（0-1），默认0.8
 * @param enable_auto_resolve 是否启用自动解决
 * @return int 成功返回0，失败返回-1
 */
int conflict_detector_set_params(ConflictDetector* detector,
                                 double spatial_threshold,
                                 double resource_threshold,
                                 int enable_auto_resolve);

// ==================== 消息加密接口 ====================

/**
 * @brief 加密类型枚举
 */
typedef enum {
    ENCRYPT_TYPE_NONE = 0,          /**< 不加密 */
    ENCRYPT_TYPE_XOR = 1,           /**< XOR流密码 */
    ENCRYPT_TYPE_XOR_ROTATE = 2     /**< XOR旋转密钥增强 */
} EncryptType;

/**
 * @brief 知识条目结构体
 */
typedef struct {
    char* entry_id;                 /**< 条目唯一标识 */
    char* knowledge_type;           /**< 知识类型（如"fact"、"rule"、"pattern"） */
    char* knowledge_data;           /**< 知识数据（JSON序列化） */
    double timestamp;               /**< 创建时间戳 */
    int version;                    /**< 版本号 */
    char* source_system_id;         /**< 来源系统ID */
} MultiKnowledgeEntry;

/**
 * @brief 知识同步状态
 */
typedef struct {
    int active_sync_count;          /**< 活动同步数量 */
    int total_entries_synced;       /**< 总同步条目数 */
    int total_entries_received;     /**< 总接收条目数 */
    char last_sync_system_id[64];   /**< 最后同步系统ID */
    double last_sync_time;          /**< 最后同步时间 */
    int is_syncing;                 /**< 是否正在同步 */
} KnowledgeSyncStatus;

/**
 * @brief 设置消息加密密钥
 * 
 * @param key 密钥数据
 * @param key_len 密钥长度
 * @return int 成功返回0，失败返回-1
 */
int multisystem_set_encryption_key(const unsigned char* key, int key_len);

/**
 * @brief 设置消息加密类型
 * 
 * @param type 加密类型
 * @return int 成功返回0，失败返回-1
 */
int multisystem_set_encryption_type(EncryptType type);

/**
 * @brief 加密消息数据
 * 
 * @param plaintext 明文数据
 * @param plaintext_len 明文长度
 * @param ciphertext 输出密文缓冲区（应分配为 plaintext_len + 16）
 * @param ciphertext_len 输出密文长度
 * @return int 成功返回0，失败返回-1
 */
int multisystem_encrypt_message(const unsigned char* plaintext, int plaintext_len,
                                unsigned char* ciphertext, int* ciphertext_len);

/**
 * @brief 解密消息数据
 * 
 * @param ciphertext 密文数据
 * @param ciphertext_len 密文长度
 * @param plaintext 输出明文缓冲区（应分配为 ciphertext_len + 1）
 * @param plaintext_len 输出明文长度
 * @return int 成功返回0，失败返回-1
 */
int multisystem_decrypt_message(const unsigned char* ciphertext, int ciphertext_len,
                                unsigned char* plaintext, int* plaintext_len);

/**
 * @brief 获取当前加密配置
 * 
 * @param type 输出加密类型
 * @return int 成功返回0，失败返回-1
 */
int multisystem_get_encryption_status(EncryptType* type);

// ==================== 跨系统知识同步接口 ====================

/**
 * @brief 启动跨系统知识同步
 * 
 * @param target_host 目标系统主机地址
 * @param target_port 目标系统端口
 * @return int 成功返回0，失败返回-1
 */
int multisystem_knowledge_sync_start(const char* target_host, int target_port);

/**
 * @brief 停止跨系统知识同步
 * 
 * @return int 成功返回0，失败返回-1
 */
int multisystem_knowledge_sync_stop(void);

/**
 * @brief 添加知识条目（本地，同步时自动分发给其他系统）
 * 
 * @param entry 知识条目
 * @return int 成功返回条目索引，失败返回-1
 */
int multisystem_knowledge_add_entry(const MultiKnowledgeEntry* entry);

/**
 * @brief 获取本地知识条目
 * 
 * @param entry_id 条目ID（NULL表示获取全部）
 * @param entries 输出条目列表（需要调用者通过 multisystem_knowledge_free_entries 释放）
 * @param entry_count 输出条目数量
 * @return int 成功返回0，失败返回-1
 */
int multisystem_knowledge_get_entries(const char* entry_id,
                                      MultiKnowledgeEntry*** entries,
                                      size_t* entry_count);

/**
 * @brief 释放知识条目列表
 * 
 * @param entries 条目列表
 * @param entry_count 条目数量
 */
void multisystem_knowledge_free_entries(MultiKnowledgeEntry** entries, size_t entry_count);

/**
 * @brief 获取知识同步状态
 * 
 * @param status 输出同步状态
 * @return int 成功返回0，失败返回-1
 */
int multisystem_knowledge_get_sync_status(KnowledgeSyncStatus* status);

/**
 * @brief 处理接收到的知识同步请求（内部回调，也开放给外部使用）
 * 
 * @param entries 接收到的知识条目
 * @param entry_count 条目数量
 * @param from_system_id 来源系统ID
 * @return int 成功返回0，失败返回-1
 */
int multisystem_knowledge_receive_entries(MultiKnowledgeEntry** entries, size_t entry_count,
                                          const char* from_system_id);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MULTISYSTEM_CONTROL_H */