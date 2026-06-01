/**
 * @file state.h
 * @brief 网络状态管理接口
 * 
 * 管理和监控液态神经网络的状态，包括稳定性、收敛性等指标。
 */

#ifndef SELFLNN_STATE_H
#define SELFLNN_STATE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 网络状态配置
 */
typedef struct {
    size_t state_size;          /**< 状态向量大小 */
    float stability_threshold;  /**< 稳定性阈值 */
    float convergence_rate;     /**< 收敛率 */
    int enable_monitoring;      /**< 是否启用监控 */
} NetworkStateConfig;

/**
 * @brief 网络状态句柄
 */
typedef struct NetworkState NetworkState;

/**
 * @brief 创建网络状态实例
 * 
 * @param config 状态配置
 * @return NetworkState* 状态句柄，失败返回NULL
 */
NetworkState* network_state_create(const NetworkStateConfig* config);

/**
 * @brief 创建网络状态实例（基础版本）
 * 
 * @param state_size 状态向量大小
 * @return NetworkState* 状态句柄，失败返回NULL
 */
NetworkState* network_state_create_simple(size_t state_size);

/**
 * @brief 释放网络状态实例
 * 
 * @param state 状态句柄
 */
void network_state_free(NetworkState* state);

/**
 * @brief 更新网络状态
 * 
 * @param state 状态句柄
 * @param new_state 新状态向量
 * @param state_size 状态向量大小
 * @return int 成功返回0，失败返回-1
 */
int network_state_update(NetworkState* state, const float* new_state, size_t state_size);

/**
 * @brief 获取网络稳定性指标
 * 
 * @param state 状态句柄
 * @param stability 稳定性输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int network_state_get_stability(const NetworkState* state, float* stability);

/**
 * @brief 获取网络收敛率
 * 
 * @param state 状态句柄
 * @param convergence 收敛率输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int network_state_get_convergence(const NetworkState* state, float* convergence);

/**
 * @brief 获取网络状态变化率
 * 
 * @param state 状态句柄
 * @param change_rate 变化率输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int network_state_get_change_rate(const NetworkState* state, float* change_rate);

/**
 * @brief 重置网络状态
 * 
 * @param state 状态句柄
 */
void network_state_reset(NetworkState* state);

/**
 * @brief 获取网络状态配置
 * 
 * @param state 状态句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int network_state_get_config(const NetworkState* state, NetworkStateConfig* config);

/**
 * @brief 设置网络状态配置
 * 
 * @param state 状态句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int network_state_set_config(NetworkState* state, const NetworkStateConfig* config);

/**
 * @brief 设置拉普拉斯频域稳定性指标
 *
 * 设置由拉普拉斯分析器计算的网络动态频域指标，包括稳定性分数、
 * 推荐的滤波截止频率和频域带宽。这些指标用于监控网络在频域的
 * 动态行为，为自适应滤波和稳定性控制提供依据。
 *
 * @param state 状态句柄
 * @param stability_score 拉普拉斯稳定性分数（0.0-1.0）
 * @param recommended_cutoff 推荐的滤波截止频率（Hz）
 * @param frequency_bandwidth 频域带宽（Hz）
 */
void network_state_set_laplace_metrics(NetworkState* state,
                                        float stability_score,
                                        float recommended_cutoff,
                                        float frequency_bandwidth);

/**
 * @brief 获取拉普拉斯稳定性分数
 *: 添加getter使LNN层可将score传播到CfCCell，
 * 补充state.c私有NetworkState结构体的读访问。
 */
float network_state_get_laplace_stability_score(const NetworkState* state);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_STATE_H