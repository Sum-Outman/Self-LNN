/**
 * @file backend.h
 * @brief 后端API接口
 * 
 * 提供HTTP API接口，允许前端与SELF-LNN系统交互。
 */

#ifndef SELFLNN_BACKEND_H
#define SELFLNN_BACKEND_H

#include <stddef.h>
#include "selflnn/core/port_config.h"
#include "selflnn/self_cognition.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 后端服务器句柄
 */
typedef struct BackendServer BackendServer;

/**
 * @brief 默认服务器端口（统一端口配置）
 * @see port_config.h SELFLNN_HTTP_PORT
 */
#define SELFLNN_DEFAULT_PORT SELFLNN_HTTP_PORT

/**
 * @brief 后端配置
 */
typedef struct {
    int port;                     /**< 服务器端口号（设为0则使用SELFLNN_DEFAULT_PORT） */
    int max_connections;          /**< 最大连接数 */
    int enable_logging;           /**< 是否启用日志记录 */
    const char* log_file;         /**< 日志文件路径 */
    int enable_multimodal;        /**< 是否启用多模态处理 */
    int enable_reasoning;         /**< 是否启用推理引擎 */
    int enable_learning;          /**< 是否启用学习功能 */
    int enable_self_evolution;    /**< 是否启用自我演化 */
    int enable_robotics;          /**< 是否启用机器人控制 */
    int enable_cognition;         /**< 是否启用自我认知系统 */
    int enable_self_decision;     /**< 是否启用自我决策 */
    int enable_self_execution;    /**< 是否启用自主执行 */
    int enable_self_learning;     /**< 是否启用自我学习 */
    int enable_self_evolution_ability; /**< 是否启用自我演化能力 */
    int enable_imitation_learning; /**< 是否启用模仿学习 */
    int enable_self_correction;   /**< 是否启用自我修正 */
    char api_key[128];            /**< API密钥（用于接口认证，空字符串表示不启用认证） */
    
    /* 限流安全配置 */
    int enable_rate_limiting;     /**< 是否启用请求限流 */
    int rate_limit_per_minute;    /**< 每分钟最大请求数（默认60） */
    int max_requests_per_minute;  /**< 每分钟最大请求数 */
    int enable_ip_whitelist;      /**< 是否启用IP白名单 */
    int enable_ip_blacklist;      /**< 是否启用IP黑名单 */
    char ip_whitelist_entries[16][64]; /**< IP白名单列表（最多16个条目） */
    int ip_whitelist_count;       /**< IP白名单条目数 */
    char ip_blacklist_entries[16][64]; /**< IP黑名单列表（最多16个条目） */
    int ip_blacklist_count;       /**< IP黑名单条目数 */
    int enable_multi_robot;       /**< 是否启用多机器人协调 */
    int enable_gpu;               /**< 是否启用GPU计算 */
    int enable_gpu_training;      /**< 是否启用GPU训练 */
    int enable_dialogue;          /**< 是否启用对话系统 */
    int enable_vision;            /**< 是否启用视觉系统 */
    int enable_slam;              /**< 是否启用SLAM */
    int enable_tts;               /**< 是否启用语音合成 */
    int enable_planning;          /**< 是否启用规划能力 */
    
    /* W-005: 线程模型与并发配置 */
    int thread_model;              /**< 线程模型: 0=单线程轮询(默认,低延迟), 1=线程池(高并发) */
    int thread_pool_size;          /**< 线程池大小(thread_model=1时生效, 默认4, 最大64) */
    int enable_async_io;           /**< 是否启用异步I/O(epoll/IOCP, 默认0=阻塞accept) */
    
    /* K-016: 安全防护配置 */
    size_t max_request_body_size; /**< 最大请求体大小（字节，默认16MB=16777216） */
    int connection_timeout_ms;    /**< 连接超时时间（毫秒，默认30000=30秒） */
    int max_connections_per_ip;   /**< 每个IP最大并发连接数（默认16，0=不限制） */
} BackendConfig;

/**
 * @brief API请求类型枚举
 */
typedef enum {
    API_NOT_FOUND = -1,           /**< 哨兵值：未匹配到任何路由 */
    API_GET_STATUS = 0,           /**< 获取系统状态 */
    API_GET_MEMORY = 1,           /**< 获取内存状态 */
    API_GET_REASONING = 2,        /**< 执行推理 */
    API_GET_LEARNING = 3,         /**< 执行学习 */
    API_POST_VISION = 4,          /**< 处理视觉输入 */
    API_POST_AUDIO = 5,           /**< 处理音频输入 */
    API_POST_TEXT = 6,            /**< 处理文本输入 */
    API_POST_SENSOR = 7,          /**< 处理传感器输入 */
    API_POST_TRAINING = 8,        /**< 训练请求 */
    API_POST_EVOLUTION = 9,       /**< 演化请求 */
    API_POST_EVOLUTION_PARETO = 96, /**< 获取帕累托前沿 */
    API_POST_RESET = 10,          /**< 重置系统 */
    API_POST_SHUTDOWN = 11,       /**< 关闭系统 */
    API_GET_ROBOT_STATUS = 12,    /**< 获取机器人状态 */
    API_POST_ROBOT_COMMAND = 13,  /**< 发送机器人控制命令 */
    API_GET_ROBOT_SENSOR = 14,    /**< 获取传感器数据 */
    API_POST_ROBOT_TRAJECTORY = 15, /**< 执行机器人轨迹 */
    API_POST_ROBOT_EMERGENCY_STOP = 16, /**< 机器人紧急停止 */
    API_POST_BACKUP = 17,         /**< 系统备份 */
    API_POST_MODEL_LOAD = 18,     /**< 模型加载 */
    API_POST_DIALOGUE = 19,       /**< 对话处理 */
    API_GET_KNOWLEDGE = 20,       /**< 查询知识库 */
    API_POST_KNOWLEDGE = 21,      /**< 添加知识到知识库 */
    API_GET_AGI_FEATURES = 22,    /**< 获取AGI功能状态 */
    API_POST_AGI_FEATURE_TOGGLE = 23, /**< 切换AGI功能 */
    API_GET_DIALOGUE_HISTORY = 24,    /**< 获取对话历史 */
    API_POST_DIALOGUE_CLEAR = 25,     /**< 清除对话历史 */
    API_POST_ROBOT_CONNECT = 26,      /**< 连接机器人 */
    API_POST_ROBOT_DISCONNECT = 27,   /**< 断开机器人连接 */
    API_GET_ROBOT_LIST = 28,          /**< 获取机器人列表(含ROS) */
    API_GET_ROS_STATUS = 29,          /**< 获取ROS Master状态 */
    API_POST_ROS_CONFIGURE = 30,      /**< 配置ROS Master连接 */
    API_GET_ROS_NODES = 31,           /**< 获取ROS节点列表 */
    API_GET_ROS_TOPICS = 32,          /**< 获取ROS主题列表 */
    API_GET_SENSOR_PIPELINE_STATUS = 33, /**< 获取传感器管道状态 */
    API_POST_GAZEBO_CONTROL = 34,     /**< Gazebo仿真控制 */
    API_POST_ROBOT_TRAINING = 35,     /**< 机器人训练控制 */
    API_POST_AUDIO_RECOGNIZE = 36,    /**< 语音识别 */
    API_POST_TTS_SYNTHESIZE = 37,     /**< 语音合成 */
    API_POST_DIALOGUE_MULTIMODAL = 38, /**< 多模态对话 */
    API_POST_AGI_EXECUTE = 39,        /**< AGI任务执行 */
    API_POST_AGI_TASK_STATUS = 40,    /**< AGI任务状态查询 */
    API_POST_DEVICE_COMMAND_V1 = 41,   /**< 设备控制命令（v1，参见API_POST_DEVICE_COMMAND=226） */
    API_GET_STATS = 42,               /**< 获取服务器统计 */
    API_POST_COMPUTER_LAUNCH = 43,    /**< 启动应用程序 */
    API_POST_COMPUTER_CLOSE = 44,     /**< 关闭应用程序 */
    API_POST_COMPUTER_TYPE = 45,      /**< 模拟键盘输入 */
    API_POST_COMPUTER_SCREENSHOT = 46, /**< 截取屏幕 */
    API_POST_COMPUTER_EXECUTE = 47,   /**< 执行系统命令 */
    API_POST_COMPUTER_VOLUME = 48,    /**< 系统音量控制 */
    API_POST_ROBOT_PARAMETERS = 49,   /**< 设置机器人参数 */
    API_POST_ROBOT_COORDINATE = 50,   /**< 机器人坐标控制 */
    API_POST_DEVICE_CONTROL = 51,     /**< 设备开关控制 */
    API_POST_FILES_READ = 52,         /**< 读取文件 */
    API_POST_FILES_WRITE = 53,        /**< 写入文件 */
    API_POST_FILES_DELETE = 54,       /**< 删除文件 */
    API_GET_FILES_LIST = 55,          /**< 列出目录文件 */
    API_POST_DEVICES_LIST = 56,       /**< 列出所有可用设备 */
    API_POST_DEVICES_REGISTER = 57,   /**< 注册前端设备 */
    API_POST_DEVICES_UNREGISTER = 58, /**< 注销前端设备 */
    API_POST_DEVICES_STATUS = 59,     /**< 获取设备状态 */
    API_POST_AUDIO_STREAM = 60,       /**< 处理实时语音流 */
    API_POST_AUDIO_COMMAND = 61,      /**< 语音指令识别与执行 */
    API_POST_VIDEO_STREAM = 62,       /**< 处理实时视频流 */
    API_POST_VIDEO_CAPTURE = 63,      /**< 视频帧捕获与处理 */
    API_POST_ROS_PUBLISH = 64,        /**< ROS话题发布 */
    API_POST_ROS_SUBSCRIBE = 65,      /**< ROS话题订阅 */
    API_POST_ROS_SERVICE = 66,        /**< ROS服务调用 */
    API_GET_SERIAL_LIST = 67,         /**< 获取串口设备列表 */
    API_POST_SERIAL_OPEN = 68,        /**< 打开串口连接 */
    API_POST_SERIAL_CLOSE = 69,       /**< 关闭串口连接 */
    API_POST_SERIAL_SEND = 70,        /**< 串口数据发送 */
    API_POST_MULTI_ROBOT_SYNC = 71,   /**< 多机器人同步协调 */
    API_GET_AGI_FEATURE_LIST = 72,    /**< 获取所有AGI功能状态列表 */
    API_POST_AGI_SELF_CORRECTION = 73, /**< 触发自我修正 */
    API_POST_KEY_SET = 74,            /**< 设置/更新API密钥 */
    API_GET_KEY_STATUS = 75,          /**< 获取API密钥状态 */
    API_POST_LEARNING_DIALOGUE = 76,  /**< 从对话中学习 */
    API_GET_API_DOCS = 77,            /**< 获取API文档 */
    API_GET_API_KEY_DOCS = 78,        /**< 获取API密钥使用文档 */
    API_POST_TRAINING_FROM_SCRATCH = 79,  /**< 从零开始训练 */
    API_POST_TRAINING_PRETRAIN = 80,      /**< 预训练 */
    API_POST_TRAINING_FINE_TUNE = 81,     /**< 微调训练 */
    API_POST_TRAINING_TRANSFER = 82,      /**< 迁移学习 */
    API_POST_TRAINING_CONTINUAL = 83,     /**< 持续学习 */
    API_POST_TRAINING_EXTERNAL_API = 84,   /**< 使用外部API训练本地模型 */
    API_POST_IMITATION_DEMONSTRATION = 85, /**< 提交模仿学习示范数据 */
    API_POST_IMITATION_TRAIN = 86,         /**< 触发模仿学习训练 */
    API_POST_IMITATION_PREDICT = 87,       /**< 模仿学习策略预测 */
    API_GET_IMITATION_STATUS = 88,         /**< 获取模仿学习状态 */
    API_POST_IMITATION_ALGORITHM = 89,     /**< 切换模仿学习算法 */
    API_POST_LEARNING_FROM_MANUAL = 90,    /**< 从说明书/教学文本学习 */
    API_GET_HEALTH = 91,                   /**< 健康检查 */
    API_GET_AGI_COGNITION_STATE = 92,       /**< 获取自我认知详细状态 */
    
    /* API密钥管理增强 */
    API_GET_KEY_LIST = 101,              /**< 获取所有API密钥列表 */
    API_POST_KEY_CREATE = 102,           /**< 创建新API密钥 */
    API_POST_KEY_DELETE = 103,           /**< 删除指定API密钥 */
    API_POST_KEY_UPDATE = 104,           /**< 更新API密钥属性 */
    API_GET_API_STATS = 105,             /**< 获取API调用统计 */
    API_GET_RATE_LIMIT_STATUS = 106,     /**< 获取速率限制状态 */
    API_POST_KEY_TOGGLE = 107,           /**< 启用/禁用API密钥 */
    
    /* 拉普拉斯频域分析 */
    API_POST_LAPLACE_SPECTRUM = 99,        /**< 获取梯度频谱分析 */
    API_POST_LAPLACE_ADAPTIVE_LR = 100,     /**< 频域自适应学习率调整 */

    /* ===== 训练中心新端点 ===== */
    API_POST_TRAINING_START = 108,         /**< 启动训练任务 */
    API_GET_TRAINING_STATUS = 109,         /**< 获取训练状态 */
    API_POST_TRAINING_PAUSE = 110,         /**< 暂停训练 */
    API_POST_TRAINING_STOP = 111,          /**< 停止训练 */
    API_GET_TRAINING_HISTORY = 112,        /**< 获取训练历史 */

    /* ===== 硬件检测新端点 ===== */
    API_POST_HARDWARE_SCAN = 113,          /**< 扫描硬件 */
    API_GET_HARDWARE_INFO = 114,           /**< 获取硬件信息 */
    API_POST_HARDWARE_CONFIG = 115,        /**< 配置计算后端 */

    /* ===== 仿真控制新端点 ===== */
    API_POST_SIMULATION_START = 116,       /**< 启动仿真 */
    API_POST_SIMULATION_STOP = 117,        /**< 停止仿真 */
    API_GET_SIMULATION_STATUS = 118,       /**< 获取仿真状态 */
    API_POST_SIMULATION_RESET = 119,       /**< 重置仿真 */
    API_POST_SIMULATION_PLAN_PATH = 120,   /**< 路径规划 */
    API_POST_SIMULATION_ROBOT_CTRL = 121,  /**< 仿真机器人控制 */
    API_POST_SIMULATION_RECONSTRUCT = 135, /**< 3D重建请求 */

    /* ===== 多模态学习新端点 ===== */
    API_POST_MULTIMODAL_LEARN = 122,       /**< 启动多模态学习 */
    API_GET_MULTIMODAL_STATUS = 123,       /**< 获取多模态状态 */
    API_POST_MULTIMODAL_TEACH = 127,       /**< 多模态教学 */
    API_POST_MULTIMODAL_TEACH_TEST = 128,  /**< 多模态教学测试 */

    /* ===== 语音控制新端点 ===== */
    API_GET_VOICE_HISTORY = 124,           /**< 获取语音命令历史 */
    API_POST_VOICE_RECOGNIZE = 136,        /**< 语音识别 */
    API_POST_VOICE_SYNTHESIZE = 137,       /**< 语音合成 */

    /* ===== 设备控制新端点 ===== */
    API_POST_DEVICES_MODE = 125,           /**< 设置设备控制模式 */
    API_POST_DEVICES_EMERGENCY_STOP = 126, /**< 紧急停止所有设备 */

    /* ===== 技能库新端点 ===== */
    API_GET_SKILLS = 130,                  /**< 获取技能库列表 */
    API_POST_SKILLS_SEARCH = 131,          /**< 搜索技能 */
    API_POST_SKILLS_EXECUTE = 132,         /**< 执行技能 */
    API_POST_SKILLS_COMPOSE = 133,         /**< 组合新技能 */
    API_GET_SKILLS_STATS = 134,            /**< 获取技能库统计 */

    /* ===== 安全监控新端点 ===== */
    API_GET_SAFETY_STATUS = 140,           /**< 获取安全监控状态 */
    API_GET_SAFETY_EVENTS = 141,           /**< 获取安全事件列表 */
    API_POST_SAFETY_EMERGENCY_STOP = 142,  /**< 紧急停止 */
    API_POST_SAFETY_SOFT_STOP = 143,       /**< 软停止 */
    API_POST_SAFETY_RESET = 144,           /**< 重置安全状态 */

    /* ===== 自主学习新端点 ===== */
    API_POST_AUTO_LEARN_SCAN = 150,        /**< 扫描知识库目录 */
    API_GET_AUTO_LEARN_STATS = 151,        /**< 获取自主学习统计 */
    API_POST_AUTO_LEARN_EXPORT = 152,      /**< 导出学习结果到知识库 */
    API_POST_AUTO_LEARN_TOGGLE = 153,      /**< 切换自主学习开关 */

    /* ===== 机器人固件/重启/校准 ===== */
    API_POST_ROBOT_FIRMWARE = 160,         /**< 刷写机器人固件 */
    API_POST_ROBOT_REBOOT = 161,           /**< 重启机器人 */
    API_POST_ROBOT_CALIBRATE = 162,        /**< 校准机器人传感器 */

    /* ===== 系统诊断 ===== */
    API_GET_SYSTEM_DIAGNOSTIC = 163,       /**< 获取系统诊断信息 */
    API_GET_SYSTEM_EXPORT_DIAGNOSTIC = 164,/**< 导出诊断数据 */

    /* ===== 多模态扩展控制 ===== */
    API_POST_MULTIMODAL_CONFIG = 165,      /**< 配置多模态参数 */
    API_POST_MULTIMODAL_PROCESS = 166,     /**< 多模态联合处理 */
    API_POST_MULTIMODAL_RESET = 167,       /**< 重置多模态系统 */
    API_POST_MULTIMODAL_STOP = 168,        /**< 停止多模态处理 */

    /* ===== 机器人扩展控制 ===== */
    API_POST_ROBOT_CONFIG_RESET = 169,     /**< 重置机器人配置 */
    API_POST_ROBOT_ANALYZE_SCREEN = 170,   /**< 分析屏幕UI元素 */
    API_POST_ROBOT_EXECUTE_ACTION = 171,   /**< 执行单个动作 */
    API_POST_ROBOT_EXECUTE_TASK = 172,     /**< 执行任务计划 */
    API_POST_ROBOT_STOP_TASK = 173,        /**< 停止当前任务 */
    API_POST_ROBOT_LEARN_FROM_DEMO = 174,  /**< 从演示中学习 */
    API_POST_ROBOT_VERIFY_ACTION = 175,    /**< 验证动作执行结果 */

    /* ===== LNN液态神经网络控制 ===== */
    API_GET_LNN_STATUS = 176,              /**< 获取LNN状态 */
    API_POST_LNN_PARAMETERS = 177,         /**< 设置LNN参数 */

    /* ===== 训练导出/日志 ===== */
    API_POST_TRAINING_EXPORT = 178,        /**< 导出训练数据 */
    API_POST_TRAINING_LOG_CLEAR = 179,     /**< 清除训练日志 */

    /* ===== GPU状态 ===== */
    API_GET_GPU_STATUS = 180,              /**< 获取GPU状态 */

    /* ===== 实物教学系统 ===== */
    API_POST_TEACH_LOOK_AND_LEARN = 181,   /**< 视觉单样本学习 */
    API_POST_TEACH_SAY_AND_ASSOCIATE = 182,/**< 语音+视觉关联学习 */
    API_POST_TEACH_TOUCH_AND_UNDERSTAND = 183, /**< 触觉传感器学习 */
    API_POST_TEACH_COUNT_AND_GENERALIZE = 184, /**< 视觉数量概念学习 */
    API_POST_TEACH_TEST_CONCEPT = 185,     /**< 测试概念识别 */
    API_GET_TEACH_GET_CONCEPTS = 186,      /**< 获取已学概念列表 */
    API_POST_TEACH_CLEAR_ALL_CONCEPTS = 187, /**< 清空所有已学概念 */
    API_POST_TEACH_CLEAR_CONCEPT = 188,    /**< 删除指定概念 */

    /* ===== 训练恢复端点 ===== */
    API_POST_TRAINING_RESUME = 189,        /**< 恢复训练 */

    /* ===== 音频/摄像头硬件采集 ===== */
    API_GET_AUDIO_DEVICES = 190,           /**< 枚举音频采集设备 */
    API_POST_AUDIO_CAPTURE_START = 191,    /**< 启动音频采集 */
    API_POST_AUDIO_CAPTURE_STOP = 192,     /**< 停止音频采集 */
    API_GET_CAMERA_DEVICES = 193,          /**< 枚举摄像头设备 */
    API_POST_CAMERA_CAPTURE_START = 194,   /**< 启动摄像头采集 */
    API_POST_CAMERA_CAPTURE_STOP = 195,    /**< 停止摄像头采集 */

    /* ===== GPU诊断增强 ===== */
    API_GET_GPU_DIAGNOSTIC = 196,          /**< GPU完整诊断 */
    API_POST_GPU_BENCHMARK = 197,          /**< GPU基准测试 */

    /* ===== 数据集管理 ===== */
    API_GET_DATASET_LIST = 198,            /**< 列出所有数据集 */
    API_POST_DATASET_CREATE = 199,         /**< 创建新数据集 */
    API_POST_DATASET_IMPORT = 200,         /**< 导入数据集 */
    API_GET_DATASET_STATS = 201,           /**< 获取数据集统计 */
    API_POST_DATASET_AUGMENT = 202,        /**< 数据增强 */

    /* ===== AGI核心能力端点(E-01) ===== */
    API_POST_AGI_THINK = 203,              /**< AGI思考：自我认知+深度思维链+深度反思 */
    API_POST_AGI_DECIDE = 204,             /**< AGI决策：决策引擎+因果推理 */
    API_POST_AGI_LEARN = 205,              /**< AGI学习：在线学习+知识整合 */
    API_POST_AGI_EVOLVE = 206,             /**< AGI进化：进化引擎+参数优化 */
    API_POST_AGI_MEMORY = 207,             /**< AGI记忆：记忆系统读写 */
    API_POST_AGI_PLAN = 208,               /**< AGI规划：分层规划+长期规划 */
    API_POST_SYSTEM_CONFIG_UPDATE = 209,    /**< 更新系统配置键值 */
    API_POST_SYSTEM_SETTINGS = 210,         /**< 保存系统设置 */
    API_POST_SYSTEM_CHANGE_PASSWORD = 211,  /**< 修改系统密码 */
    API_GET_HARDWARE_RESOURCES = 212,       /**< 获取硬件资源分配 */
    API_POST_HARDWARE_RESOURCES_ALLOCATE = 213, /**< 自动分配硬件资源 */

    /* ===== 自我编程端点 ===== */
    API_POST_PROGRAMMING_ANALYZE = 214,    /**< 分析代码复杂度与质量 */
    API_POST_PROGRAMMING_GENERATE = 215,   /**< 生成代码 */
    API_POST_PROGRAMMING_OPTIMIZE = 216,   /**< 优化代码 */
    API_POST_PROGRAMMING_COMPILE = 217,    /**< 编译验证 */
    API_POST_PROGRAMMING_EXECUTE = 218,    /**< 沙箱执行 */
    API_GET_PROGRAMMING_STATUS = 219,      /**< 获取编程引擎状态 */

    /* ===== K-023: 能力开关诊断 (实际dispatch使用260-266) ===== */
    /* 注意：以下枚举值220-229在dispatch中被知识库/记忆路由复用，
     * 能力开关/设备发现实际使用260-266号槽位，此处仅保留枚举名称引用 */
    API_GET_CAPABILITY_DIAGNOSE = 260,     /**< 12项能力开关完整状态诊断 */

    /* ===== K-016: 设备发现与多系统控制 ===== */
    API_GET_DEVICES_DISCOVER = 261,        /**< 扫描发现所有可用设备 */
    API_POST_DEVICES_DISCOVER = 262,       /**< 触发设备扫描 */
    API_POST_DEVICE_REGISTER = 263,        /**< 注册新设备 */
    API_POST_DEVICE_UNREGISTER = 264,      /**< 注销设备 */
    API_GET_DEVICE_LIST = 265,             /**< 获取已注册设备列表 */
    API_POST_DEVICE_COMMAND = 266,         /**< 向设备发送控制命令（注意：区别于API_POST_DEVICE_COMMAND_V1=41） */
    API_GET_DEVICE_STATUS = 267,            /**< 获取设备实时状态 */
    API_GET_SAFETY_BOUNDS = 228            /**< 获取安全边界配置 */
} ApiRequestType;

/**
 * @brief API密钥权限级别
 */
typedef enum {
    API_KEY_PERM_READONLY = 0,      /**< 只读：仅允许GET请求 */
    API_KEY_PERM_READWRITE = 1,     /**< 读写：允许GET和POST请求 */
    API_KEY_PERM_ADMIN = 2          /**< 管理：允许所有操作，包括密钥管理 */
} ApiKeyPermission;

/**
 * @brief API密钥条目
 */
typedef struct {
    char key_value[128];             /**< 密钥值 */
    char name[64];                   /**< 密钥名称/备注 */
    ApiKeyPermission permission;     /**< 权限级别 */
    int enabled;                     /**< 是否启用 */
    time_t created_at;               /**< 创建时间 */
    time_t expires_at;               /**< 过期时间（0=永不过期） */
    time_t last_used_at;             /**< 最后使用时间 */
    int usage_count;                 /**< 使用次数 */
} ApiKeyEntry;

/**
 * @brief API统计数据
 */
typedef struct {
    int total_requests;              /**< 总请求数 */
    int current_keys;                /**< 当前密钥数 */
    int enabled_keys;                /**< 已启用密钥数 */
    int requests_last_hour;          /**< 最近一小时请求数 */
    int rate_limit_remaining;        /**< 速率限制剩余额度 */
    int active_connections;          /**< 当前活跃连接数 */
    int error_count;                 /**< 错误计数 */
    double avg_response_time_ms;     /**< 平均响应时间（毫秒） */
} ApiStats;

/**
 * @brief API响应结构
 */
typedef struct {
    int status_code;              /**< 状态码：200=成功，400=错误，500=服务器错误 */
    char* content_type;           /**< 内容类型：application/json, text/plain */
    char* data;                   /**< 响应数据（JSON格式） */
    size_t data_length;           /**< 数据长度 */
} ApiResponse;

/**
 * @brief 默认配置文件路径
 */
#define SELFLNN_CONFIG_FILE "selflnn_config.json"

/**
 * @brief 保存后端配置到文件（JSON格式）
 *
 * 将 BackendConfig 中所有开关状态和配置项持久化到磁盘。
 * 保存路径依次使用以下优先级：
 *   1. backend_save_config() 的 path 参数（非NULL）
 *   2. SELFLNN_CONFIG_PATH 环境变量
 *   3. 默认值 SELFLNN_CONFIG_FILE
 *
 * @param config 要保存的配置（若为NULL则保存时为当前值）
 * @param path 保存路径（NULL则使用自动检测路径）
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int backend_save_config(const BackendConfig* config, const char* path);

/**
 * @brief 从文件加载后端配置（JSON格式）
 *
 * 从磁盘读取 JSON 配置文件并填充 BackendConfig。
 * 文件中不存在的字段保持原始值不变。
 *
 * @param config [out] 接收加载的配置
 * @param path 加载路径（NULL则使用自动检测路径）
 * @return int 成功返回0，失败返回-1（文件不存在返回1）
 */
SELFLNN_API int backend_load_config(BackendConfig* config, const char* path);

/**
 * @brief 创建后端服务器
 * 
 * @param config 后端配置
 * @return BackendServer* 后端服务器句柄，失败返回NULL
 */
BackendServer* backend_server_create(const BackendConfig* config);

/**
 * @brief 启动后端服务器
 * 
 * @param server 后端服务器句柄
 * @return int 成功返回0，失败返回-1
 */
int backend_server_start(BackendServer* server);

/**
 * @brief 停止后端服务器
 * 
 * @param server 后端服务器句柄
 * @return int 成功返回0，失败返回-1
 */
int backend_server_stop(BackendServer* server);

/**
 * @brief 释放后端服务器
 * 
 * @param server 后端服务器句柄
 */
void backend_server_free(BackendServer* server);

/**
 * @brief 处理API请求
 * 
 * @param server 后端服务器句柄
 * @param request_type 请求类型
 * @param request_data 请求数据（JSON格式）
 * @param request_length 请求数据长度
 * @return ApiResponse* API响应，调用者需要释放
 */
ApiResponse* backend_handle_request(BackendServer* server,
                                   ApiRequestType request_type,
                                   const char* request_data,
                                   size_t request_length,
                                   const char* client_addr);

/**
 * @brief 释放API响应
 * 
 * @param response API响应
 */
void backend_response_free(ApiResponse* response);

/**
 * @brief 获取服务器运行状态
 * 
 * @param server 后端服务器句柄
 * @return int 1=运行中，0=停止，-1=错误
 */
int backend_server_is_running(const BackendServer* server);

/**
 * @brief 获取服务器统计信息
 * 
 * @param server 后端服务器句柄
 * @return char* JSON格式的统计信息，调用者需要释放
 */
char* backend_server_get_stats(const BackendServer* server);

/**
 * @brief 获取服务器健康检查信息
 * 
 * 返回包含服务器状态、运行时间、连接数、GPU/CPU信息、组件状态等的JSON数据
 * 
 * @param server 后端服务器句柄
 * @return char* JSON格式的健康检查信息，调用者需要释放
 */
char* backend_server_get_health(const BackendServer* server);

/**
 * @brief 统一启用或禁用后端服务器的指定AGI核心功能
 *
 * 通过FeatureType枚举统一管理所有核心功能的运行时启停。
 * 自动同步到认知系统的feature_states。
 * 替换各个模块分散的enable/disable开关，提供单一入口。
 *
 * @param server 后端服务器句柄
 * @param feature 要启停的功能类型
 * @param enable 1=启用，0=禁用
 * @return int 成功返回0，失败返回-1
 */
int backend_server_enable_feature(BackendServer* server,
                                  FeatureType feature,
                                  int enable);

/**
 * @brief 查询指定功能当前是否启用
 *
 * @param server 后端服务器句柄
 * @param feature 要查询的功能类型
 * @return int 启用返回1，禁用返回0，失败返回-1
 */
int backend_server_is_feature_enabled(BackendServer* server,
                                      FeatureType feature);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_BACKEND_H */