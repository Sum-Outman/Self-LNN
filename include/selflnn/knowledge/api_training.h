/**
 * @file api_training.h
 * @brief API训练接口
 * 
 * 通过API进行模型训练的接口。
 */

#ifndef SELFLNN_API_TRAINING_H
#define SELFLNN_API_TRAINING_H

#include <stddef.h>
#include "selflnn/training/training.h"
#include "selflnn/core/lnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief OpenAI API提供者类型
 */
typedef enum {
    OPENAI_PROVIDER_OPENAI = 0,          /**< OpenAI官方API */
    OPENAI_PROVIDER_AZURE = 1,           /**< Azure OpenAI */
    OPENAI_PROVIDER_CUSTOM = 2,          /**< 自定义兼容API */
    OPENAI_PROVIDER_LOCAL = 3            /**< 本地模型API */
} OpenAIProviderType;

/**
 * @brief OpenAI兼容适配器配置
 */
typedef struct {
    OpenAIProviderType provider;         /**< 提供者类型 */
    char api_base[256];                  /**< API基础URL */
    char api_key[512];                   /**< API密钥 */
    char model[128];                     /**< 模型名称 */
    float temperature;                   /**< 温度参数 */
    float top_p;                         /**< Top-P采样 */
    int max_tokens;                      /**< 最大生成令牌数 */
    float presence_penalty;              /**< 存在惩罚 */
    float frequency_penalty;             /**< 频率惩罚 */
    int timeout_seconds;                 /**< 超时时间 */
    int max_retries;                     /**< 最大重试次数 */
    char stop_sequences[4][128];         /**< 停止序列 */
    int stop_count;                      /**< 停止序列数量 */
    int stream_enabled;                  /**< 是否启用流式响应 */
    char proxy_url[256];                 /**< 代理URL（可选） */
} OpenAIConfig;

/**
 * @brief OpenAI API响应
 */
typedef struct {
    char* content;                       /**< 响应内容 */
    size_t content_length;               /**< 内容长度 */
    char* finish_reason;                 /**< 结束原因 */
    int prompt_tokens;                   /**< 提示令牌数 */
    int completion_tokens;               /**< 完成令牌数 */
    int success;                         /**< 是否成功 */
    int http_status;                     /**< HTTP状态码 */
    char error_message[512];             /**< 错误消息 */
} OpenAIResponse;

/**
 * @brief OpenAI适配器句柄
 */
typedef struct OpenAIAdapter OpenAIAdapter;

/**
 * @brief 创建OpenAI适配器
 *
 * @param config OpenAI配置
 * @return OpenAIAdapter* 适配器句柄，失败返回NULL
 */
OpenAIAdapter* openai_adapter_create(const OpenAIConfig* config);

/**
 * @brief 释放OpenAI适配器
 *
 * @param adapter 适配器句柄
 */
void openai_adapter_free(OpenAIAdapter* adapter);

/**
 * @brief 发送聊天补全请求
 *
 * @param adapter 适配器句柄
 * @param messages JSON格式消息数组字符串
 * @param response 响应输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int openai_adapter_chat_completion(OpenAIAdapter* adapter,
                                    const char* messages,
                                    OpenAIResponse* response);

/**
 * @brief 发送训练请求（微调API）
 *
 * @param adapter 适配器句柄
 * @param training_data_json JSON格式训练数据
 * @param response 响应输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int openai_adapter_fine_tune(OpenAIAdapter* adapter,
                              const char* training_data_json,
                              OpenAIResponse* response);

/**
 * @brief 获取可用模型列表
 *
 * @param adapter 适配器句柄
 * @param response 响应输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int openai_adapter_list_models(OpenAIAdapter* adapter,
                                OpenAIResponse* response);

/**
 * @brief 释放OpenAI响应资源
 *
 * @param response 响应对象
 */
void openai_response_free(OpenAIResponse* response);

/**
 * @brief 训练状态
 */
typedef enum {
    TRAINING_STATUS_IDLE = 0,        /**< 空闲 */
    TRAINING_STATUS_RUNNING = 1,     /**< 运行中 */
    TRAINING_STATUS_PAUSED = 2,      /**< 暂停 */
    TRAINING_STATUS_COMPLETED = 3,   /**< 完成 */
    TRAINING_STATUS_FAILED = 4       /**< 失败 */
} TrainingStatus;

/**
 * @brief 训练类型
 */
typedef enum {
    TRAINING_TYPE_SUPERVISED = 0,    /**< 监督学习 */
    TRAINING_TYPE_UNSUPERVISED = 1,  /**< 无监督学习 */
    TRAINING_TYPE_REINFORCEMENT = 2, /**< 强化学习 */
    TRAINING_TYPE_EVOLUTIONARY = 3   /**< 进化学习 */
} TrainingType;

/**
 * @brief 训练配置
 * 使用selflnn/training/training.h中的TrainingConfig定义
 */
/* 原定义已移至training.h
typedef struct {
    TrainingType type;               // 训练类型
    int max_iterations;              // 最大迭代次数
    float learning_rate;             // 学习率
    float target_accuracy;           // 目标准确率
    int enable_validation;           // 是否启用验证
    int validation_split;            // 验证分割比例 (百分比)
} TrainingConfig;
*/

/**
 * @brief 训练状态信息
 */
typedef struct {
    TrainingStatus status;           /**< 训练状态 */
    int current_iteration;           /**< 当前迭代次数 */
    float current_accuracy;          /**< 当前准确率 */
    float current_loss;              /**< 当前损失 */
    float progress_percentage;       /**< 进度百分比 */
    char* message;                   /**< 状态消息 */
} TrainingStateInfo;

/**
 * @brief 训练结果
 */
typedef struct {
    int total_iterations;            /**< 总迭代次数 */
    float final_accuracy;            /**< 最终准确率 */
    float final_loss;                /**< 最终损失 */
    long training_time_ms;           /**< 训练时间（毫秒） */
    char* model_data;                /**< 模型数据 */
    size_t model_data_size;          /**< 模型数据大小 */
} TrainingResult;

/**
 * @brief 训练会话句柄
 */
typedef struct TrainingSession TrainingSession;

/**
 * @brief 创建训练会话
 * 
 * @param config 训练配置
 * @return TrainingSession* 训练会话句柄，失败返回NULL
 */
TrainingSession* training_session_create(const TrainingConfig* config);

/**
 * @brief 释放训练会话
 * 
 * @param session 训练会话句柄
 */
void training_session_free(TrainingSession* session);

/**
 * @brief 启动训练
 * 
 * @param session 训练会话句柄
 * @param training_data 训练数据
 * @param data_size 数据大小
 * @return int 成功返回0，失败返回-1
 */
int training_session_start(TrainingSession* session, 
                          const void* training_data, size_t data_size);

/**
 * @brief 暂停训练
 * 
 * @param session 训练会话句柄
 * @return int 成功返回0，失败返回-1
 */
int training_session_pause(TrainingSession* session);

/**
 * @brief 恢复训练
 * 
 * @param session 训练会话句柄
 * @return int 成功返回0，失败返回-1
 */
int training_session_resume(TrainingSession* session);

/**
 * @brief 停止训练
 * 
 * @param session 训练会话句柄
 * @return int 成功返回0，失败返回-1
 */
int training_session_stop(TrainingSession* session);

/**
 * @brief 获取训练状态
 * 
 * @param session 训练会话句柄
 * @param state 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int training_session_get_state(TrainingSession* session, TrainingStateInfo* state);

/**
 * @brief 获取训练结果
 * 
 * @param session 训练会话句柄
 * @param result 结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int training_session_get_result(TrainingSession* session, TrainingResult* result);

/**
 * @brief 导出模型
 * 
 * @param session 训练会话句柄
 * @param model_data 模型数据输出缓冲区
 * @param max_size 最大大小
 * @return int 成功返回模型数据大小，失败返回-1
 */
int training_session_export_model(TrainingSession* session, 
                                 void* model_data, size_t max_size);

/**
 * @brief 导入模型
 * 
 * @param session 训练会话句柄
 * @param model_data 模型数据
 * @param data_size 数据大小
 * @return int 成功返回0，失败返回-1
 */
int training_session_import_model(TrainingSession* session,
                                 const void* model_data, size_t data_size);

/**
 * @brief 设置训练回调函数
 * 
 * @param session 训练会话句柄
 * @param callback 回调函数指针
 * @param user_data 用户数据
 * @return int 成功返回0，失败返回-1
 */
int training_session_set_callback(TrainingSession* session,
                                 void (*callback)(TrainingStateInfo*, void*),
                                 void* user_data);

/**
 * @brief 设置要训练的神经网络
 * 
 * @param session 训练会话句柄
 * @param network 神经网络指针
 * @return int 成功返回0，失败返回-1
 */
int training_session_set_network(TrainingSession* session, LNN* network);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_API_TRAINING_H