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

/* M-001修复: LaplaceAIConfig完整结构体定义，与LaplaceConfig字段完全一致。
 * 消除不安全的强制类型转换 (const LaplaceConfig*)cfg。
 * 原为不透明指针 typedef struct LaplaceAIConfig LaplaceAIConfig;
 * 现在提供13个字段的完整定义，确保类型安全。 */
typedef struct LaplaceAIConfig {
    size_t num_samples;         /* 采样点数 */
    float sample_rate;          /* 采样率 (Hz) */
    float max_frequency;        /* 最大分析频率 (Hz) */
    float min_frequency;        /* 最小分析频率 (Hz) */
    int enable_stability;       /* 是否启用稳定性分析 */
    size_t buffer_size;         /* 内部缓冲区大小（字节） */
    int enable_frequency;       /* 是否启用频域分析 */
    int enable_optimization;    /* 是否启用优化 */
    float cutoff_frequency;     /* 截止频率 (Hz) */
    int filter_order;           /* 滤波器阶数 */
    float alpha;                /* 滤波器参数 alpha */
    float beta;                 /* 滤波器参数 beta */
    float frequency_range;      /* 频率范围 */
    int enable_auto_tuning;     /* 是否启用自动调谐 */
    float stability_threshold;  /* 稳定性阈值 */
} LaplaceAIConfig;

#ifdef __cplusplus
extern "C" {
#endif

/* 统一入口宏 —— 将分散在各子模块中的函数映射到统一命名空间
 * M-001修复: cfg直接以LaplaceAIConfig*传入，字段结构与LaplaceConfig完全相同，
 * 通过指针类型转换为(const LaplaceConfig*)安全传递（结构体内存布局一致） */
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

/* ZSFUSA: 创建默认拉普拉斯分析器 */
void* lnn_laplace_create_default_analyzer(void);

/* ZSFUSA: 获取拉普拉斯频谱 */
int laplace_unified_get_spectrum(void* analyzer, float* spectrum, size_t size);

/* H-001修复: 统一分析入口 —— 分析LNN并缓存真实传递函数
 * 调用 laplace_analyze_lnn_stability 完成分析后自动缓存传递函数系数，
 * 使后续 laplace_unified_get_spectrum 能够计算基于真实系统动态的频谱。
 * @param analyzer 拉普拉斯分析器指针
 * @param lnn 液态神经网络指针
 * @param result 稳定性分析结果输出(可为NULL)
 * @return 0成功, 非0失败 */
int laplace_unified_analyze(void* analyzer, LNN* lnn, StabilityAnalysis* result);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LAPLACE_UNIFIED_H */
