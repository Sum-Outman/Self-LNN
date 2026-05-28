/**
 * @file laplace_unified.h
 * @brief 拉普拉斯变换统一入口 —— 聚合三个拉普拉斯子模块
 *
 * 本头文件统一聚合以下模块的完整API：
 * - laplace.h — 拉普拉斯变换分析、系统稳定性评估、频域优化
 * - laplace_fft.h — FFT频域卷积、传递函数分析、功率谱密度
 * - laplace_features.h — 拉普拉斯金字塔、图拉普拉斯、频谱特征映射
 *
 * ZSFZS-F030: laplace_integration.h已删除(纯转发空头文件,无自有内容)
 * 其功能(CfC稳定性/分数阶记忆/RLS/PID/门控)已由laplace.h和laplace_unified.h覆盖。 */

#ifndef SELFLNN_LAPLACE_UNIFIED_H
#define SELFLNN_LAPLACE_UNIFIED_H

#include "selflnn/core/common.h"
#include "selflnn/core/laplace.h"
/* ZSFZS-F030: laplace_integration.h已删除(纯转发空头文件) */
/* #include "selflnn/core/laplace_ai_framework.h" */ /* 已由上述模块完全覆盖 */
/* #include "selflnn/core/laplace_enhanced.h" */     /* 已由上述模块完全覆盖 */
/* #include "selflnn/core/laplace_integration.h" */  /* ZSFZS-F030: 已删除 */

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
