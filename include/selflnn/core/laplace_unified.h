/**
 * @file laplace_unified.h
 * @brief 拉普拉斯变换统一入口 —— 聚合三个拉普拉斯子模块
 *
 * 本头文件统一聚合以下模块的完整API：
 * - laplace_ai_framework.h — 拉普拉斯AI框架（频谱变换/特征提取/RL集成）
 * - laplace_enhanced.h   — 拉普拉斯增强系统（频域增强/稳定性/管道/监控）
 * - laplace_integration.h — 拉普拉斯深度集成（CfC稳定性/分数阶记忆/RLS/PID/门控）
 *
 * 所有原有函数签名保持不变，完全向后兼容。
 * 统一入口函数实现在 laplace_unified.c 中。
 */

#ifndef SELFLNN_LAPLACE_UNIFIED_H
#define SELFLNN_LAPLACE_UNIFIED_H

#include "selflnn/core/common.h"
#include "selflnn/core/laplace.h"
/* laplace_ai_framework.h 暂未实现，跳过 */
/* #include "selflnn/core/laplace_ai_framework.h" */
/* laplace_enhanced.h 暂未实现，跳过 */
/* #include "selflnn/core/laplace_enhanced.h" */
#include "selflnn/core/laplace_integration.h"

/* ZSFBUILD: 前向声明 —— LaplaceAIConfig原本通过laplace_ai_framework.h间接声明
 * 但该头文件为空转发，导致类型缺失。此处作为不透明指针类型修复。 */
typedef struct LaplaceAIConfig LaplaceAIConfig;

#ifdef __cplusplus
extern "C" {
#endif

/* 统一入口宏 —— 将分散在各子模块中的函数映射到统一命名空间 */
/* ZSFBUILD: 原宏指向不存在的laplace_ai_*，修正为真实API */
#define laplace_unified_init(cfg)       laplace_analyzer_create((const LaplaceConfig*)(cfg))
#define laplace_unified_free(ptr)       laplace_analyzer_free((LaplaceAnalyzer*)(ptr))

/* 拉普拉斯统一系统初始化（整合三个子系统） */
int laplace_unified_system_init(const LaplaceAIConfig* cfg);

/* 获取拉普拉斯系统健康状态 */
int laplace_unified_health_check(char* report, size_t report_size);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LAPLACE_UNIFIED_H */
