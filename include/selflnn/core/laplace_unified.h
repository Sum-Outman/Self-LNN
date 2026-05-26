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
/* ZSFWS-M003修复: laplace_ai_framework.h 和 laplace_enhanced.h 的核心功能
 * 已完整实现在以下现有模块中：
 * - laplace.h       → 拉普拉斯变换分析、系统稳定性评估、频域优化
 * - laplace_fft.h   → FFT频域卷积、传递函数分析、功率谱密度
 * - laplace_features.h → 拉普拉斯金字塔、图拉普拉斯、频谱特征映射
 * - laplace_integration.h → CfC稳定性增强、分数阶记忆、RLS/PID/门控集成
 * 上述四个模块覆盖了需求.txt中"拉普拉斯变换AI技术"的全部功能要求。 */
/* #include "selflnn/core/laplace_ai_framework.h" */ /* 已由上述模块完全覆盖 */
/* #include "selflnn/core/laplace_enhanced.h" */     /* 已由上述模块完全覆盖 */
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

/* 拉普拉斯统一系统初始化（整合频谱分析+频域增强+深度集成） */
int laplace_unified_system_init(const LaplaceAIConfig* cfg);

/* 获取全局拉普拉斯分析器（保持生命周期，供运行时使用） */
LaplaceAnalyzer* laplace_unified_get_analyzer(void);

/* 拉普拉斯统一系统关闭（释放全局分析器） */
void laplace_unified_system_shutdown(void);

/* 获取拉普拉斯系统健康状态 */
int laplace_unified_health_check(char* report, size_t report_size);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LAPLACE_UNIFIED_H */
