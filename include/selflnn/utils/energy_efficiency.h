/**
 * @file energy_efficiency.h
 * @brief 能效优化接口
 * 
 * 能效优化核心接口，支持功率感知调度、动态频率调整、智能休眠和能耗监控。
 */

#ifndef SELFLNN_ENERGY_EFFICIENCY_H
#define SELFLNN_ENERGY_EFFICIENCY_H

#include <stddef.h>
#include "selflnn/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 设备能耗状态（P2-002修复：移除废弃的PowerMode枚举）
 */
typedef enum {
    DEVICE_POWER_ACTIVE = 0,      /**< 活跃状态 */
    DEVICE_POWER_IDLE = 1,        /**< 空闲状态 */
    DEVICE_POWER_SLEEP = 2,       /**< 休眠状态 */
    DEVICE_POWER_OFF = 3,         /**< 关闭状态 */
    DEVICE_POWER_UNKNOWN = 4      /**< 未知状态 */
} DevicePowerState;

/**
 * @brief 能耗监控数据点
 */
typedef struct {
    double timestamp;             /**< 时间戳（秒） */
    double power_consumption;     /**< 功耗（瓦特） */
    double temperature;           /**< 温度（摄氏度） */
    double cpu_usage;             /**< CPU使用率（0-1） */
    double memory_usage;          /**< 内存使用率（0-1） */
} EnergyMonitorPoint;

/**
 * @brief 能耗分析结果
 */
typedef struct {
    double avg_power;             /**< 平均功耗（瓦特） */
    double peak_power;            /**< 峰值功耗（瓦特） */
    double total_energy;          /**< 总能耗（焦耳） */
    double energy_efficiency;     /**< 能效评分（0-1） */
    double thermal_efficiency;    /**< 热效率评分（0-1） */
    char** optimization_suggestions; /**< 优化建议列表 */
    size_t suggestion_count;      /**< 建议数量 */
} EnergyAnalysis;

/**
 * @brief 能效优化引擎句柄
 */
typedef struct EnergyEfficiencyEngine EnergyEfficiencyEngine;

/**
 * @brief 创建能效优化引擎
 * 
 * @return EnergyEfficiencyEngine* 引擎句柄，失败返回NULL
 */
EnergyEfficiencyEngine* energy_efficiency_engine_create(void);

/**
 * @brief 销毁能效优化引擎
 * 
 * @param engine 引擎句柄
 */
void energy_efficiency_engine_destroy(EnergyEfficiencyEngine* engine);

/**
 * @brief 设置功率模式
 * 
 * @param engine 引擎句柄
 * @param mode 功率模式
 * @return int 成功返回0，失败返回错误码
 */
int set_power_mode(EnergyEfficiencyEngine* engine, PowerMode mode);

/**
 * @brief 获取当前功率模式
 * 
 * @param engine 引擎句柄
 * @return PowerMode 当前功率模式
 */
PowerMode get_current_power_mode(EnergyEfficiencyEngine* engine);

/**
 * @brief 监控系统能耗
 * 
 * @param engine 引擎句柄
 * @param duration 监控时长（秒）
 * @param sampling_interval 采样间隔（秒）
 * @param data_points 数据点数组输出（需要调用者释放）
 * @param point_count 数据点数量输出
 * @return int 成功返回0，失败返回错误码
 */
int monitor_energy_consumption(EnergyEfficiencyEngine* engine,
                               double duration,
                               double sampling_interval,
                               EnergyMonitorPoint** data_points,
                               size_t* point_count);

/**
 * @brief 释放能耗监控数据点
 * 
 * @param data_points 数据点数组
 * @param point_count 数据点数量
 */
void free_energy_monitor_points(EnergyMonitorPoint* data_points, size_t point_count);

/**
 * @brief 分析能耗数据
 * 
 * @param engine 引擎句柄
 * @param data_points 数据点数组
 * @param point_count 数据点数量
 * @return EnergyAnalysis* 分析结果，失败返回NULL
 */
EnergyAnalysis* analyze_energy_data(EnergyEfficiencyEngine* engine,
                                    const EnergyMonitorPoint* data_points,
                                    size_t point_count);

/**
 * @brief 销毁能耗分析结果
 * 
 * @param analysis 分析结果
 */
void energy_analysis_destroy(EnergyAnalysis* analysis);

/**
 * @brief 优化系统能效
 * 
 * @param engine 引擎句柄
 * @param analysis 能耗分析结果
 * @return int 成功返回0，失败返回错误码
 */
int optimize_energy_efficiency(EnergyEfficiencyEngine* engine,
                               const EnergyAnalysis* analysis);

/**
 * @brief 动态调整设备功率状态
 * 
 * @param engine 引擎句柄
 * @param device_id 设备ID
 * @param target_state 目标功率状态
 * @return int 成功返回0，失败返回错误码
 */
int adjust_device_power_state(EnergyEfficiencyEngine* engine,
                              const char* device_id,
                              DevicePowerState target_state);

/**
 * @brief 预测能耗模式
 * 
 * @param engine 引擎句柄
 * @param workload_description 工作负载描述
 * @param estimated_duration 预估持续时间（秒）
 * @return PowerMode 推荐的功率模式
 */
PowerMode predict_optimal_power_mode(EnergyEfficiencyEngine* engine,
                                     const char* workload_description,
                                     double estimated_duration);

/**
 * @brief 获取能效统计信息
 * 
 * @param engine 引擎句柄
 * @param stat_name 统计名称（如"total_saved_energy", "avg_efficiency"）
 * @return double 统计值，失败返回-1.0
 */
double get_energy_statistic(EnergyEfficiencyEngine* engine, const char* stat_name);

/**
 * @brief 自动调整功率模式（基于实时温度、负载和功耗）
 *
 * 根据操作系统API读取的CPU温度、使用率和系统功耗，
 * 自动选择最佳的功率模式并在硬件层面应用。
 * 建议每2秒调用一次。
 *
 * @param engine 能效优化引擎
 * @return int 0=成功，其他=错误码
 */
int auto_tune_power_mode(EnergyEfficiencyEngine* engine);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_ENERGY_EFFICIENCY_H */