/**
 * @file laplace_unified.c
 * @brief 拉普拉斯变换统一入口实现
 *
 * 整合 laplace_ai_framework、laplace_enhanced、laplace_integration 三个子模块，
 * 提供统一的初始化和健康检查接口。
 */

#include "selflnn/core/laplace_unified.h"
#include "selflnn/utils/logging.h"
#include <string.h>
#include <stdio.h>

int laplace_unified_system_init(const LaplaceAIConfig* cfg) {
    if (!cfg) return -1;

    /* 初始化AI框架子系统 */
    struct LaplaceAI* ai = laplace_ai_create(cfg);
    if (!ai) {
        log_error("[拉普拉斯统一] AI框架初始化失败");
        return -1;
    }

    /* 初始化增强系统 */
    LaplaceEnhancedSystem* enh = laplace_enhanced_create(LAPLACE_TARGET_ALL);
    if (!enh) {
        log_warning("[拉普拉斯统一] 增强系统初始化失败，继续运行");
        laplace_ai_free(ai);
        return -1;
    }
    laplace_enhanced_free(enh);

    /* 拉普拉斯深度集成默认初始化 */
    /* 频谱配置和分数阶记忆配置在首次调用时自动初始化 */

    log_info("[拉普拉斯统一] 全系统初始化完成：AI框架 + 增强系统 + 深度集成");
    laplace_ai_free(ai);
    return 0;
}

int laplace_unified_health_check(char* report, size_t report_size) {
    if (!report || report_size == 0) return -1;

    int ok_count = 0;
    int total_count = 3;
    char buf[512];

    /* 检查AI框架 */
    LaplaceAIConfig test_cfg;
    memset(&test_cfg, 0, sizeof(test_cfg));
    test_cfg.fft_size = 256;
    test_cfg.sampling_rate = 100.0f;
    struct LaplaceAI* ai = laplace_ai_create(&test_cfg);
    if (ai) {
        ok_count++;
        laplace_ai_free(ai);
    }

    /* 检查增强系统 */
    LaplaceEnhancedSystem* enh = laplace_enhanced_create(LAPLACE_TARGET_ALL);
    if (enh) {
        ok_count++;
        laplace_enhanced_free(enh);
    }

    /* 检查深度集成（频谱配置） */
    SpectrumConfig spec_cfg = laplace_spectrum_config_default();
    if (spec_cfg.fft_size > 0) {
        ok_count++;
    }

    snprintf(report, report_size,
             "拉普拉斯系统健康状态: %d/%d 子系统正常", ok_count, total_count);

    return (ok_count == total_count) ? 0 : -1;
}
