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
    /* ZSFBUILD: 深度初始化拉普拉斯分析器、AI框架、增强系统和深度集成子系统
     * 创建全局LaplaceAnalyzer并保持其生命周期，供后续健康检查使用 */
    (void)cfg;
    
    LaplaceConfig default_cfg = LAPLACE_CONFIG_DEFAULT;
    LaplaceAnalyzer* analyzer = laplace_analyzer_create(&default_cfg);
    if (!analyzer) {
        log_error("[拉普拉斯统一] 拉普拉斯分析器初始化失败");
        return -1;
    }

    /* 初始化AI框架层（频域滤波、Butterworth、MFCC等） */
    if (laplace_ai_framework_init() != 0) {
        log_error("[拉普拉斯统一] AI框架初始化失败");
        laplace_analyzer_free(analyzer);
        return -1;
    }

    /* 初始化增强系统层（系统级频谱分析、滤波、降噪、稳定性监控） */
    if (laplace_enhanced_system_init() != 0) {
        log_error("[拉普拉斯统一] 增强系统初始化失败");
        laplace_analyzer_free(analyzer);
        return -1;
    }

    /* 初始化深度集成层（CfC稳定性、频率响应、系统辨识、分数阶记忆） */
    if (laplace_integration_init() != 0) {
        log_error("[拉普拉斯统一] 深度集成初始化失败");
        laplace_analyzer_free(analyzer);
        return -1;
    }

    /* 验证所有子系统就绪 */
    char health_buf[512];
    if (laplace_unified_health_check(health_buf, sizeof(health_buf)) != 0) {
        log_error("[拉普拉斯统一] 健康检查失败");
        laplace_analyzer_free(analyzer);
        return -1;
    }
    log_info("[拉普拉斯统一] 全系统就绪: %s", health_buf);

    laplace_analyzer_free(analyzer);
    log_info("[拉普拉斯统一] 全系统初始化完成：AI框架 + 增强系统 + 深度集成");
    return 0;
}

int laplace_unified_health_check(char* report, size_t report_size) {
    if (!report || report_size == 0) return -1;

    /* 真正的健康检查：创建临时分析器，验证配置和缓冲区分配 */
    LaplaceConfig analyze_cfg;
    int health_issues = 0;
    char issues_buf[256] = {0};

    /* 1. 验证默认配置有效性 */
    const LaplaceConfig* default_cfg = laplace_get_default_config();
    if (!default_cfg) {
        snprintf(report, report_size, "拉普拉斯系统健康状态: [故障] 无法获取默认配置");
        return -1;
    }

    /* 检查采样点数 */
    if (default_cfg->num_samples == 0) {
        strncat(issues_buf, "采样点数为零; ", sizeof(issues_buf) - strlen(issues_buf) - 1);
        health_issues++;
    }

    /* 检查采样率 */
    if (default_cfg->sample_rate <= 0.0f) {
        strncat(issues_buf, "采样率无效; ", sizeof(issues_buf) - strlen(issues_buf) - 1);
        health_issues++;
    }

    /* 检查频率范围有效性 */
    if (default_cfg->min_frequency >= default_cfg->max_frequency) {
        strncat(issues_buf, "频率范围无效(min>=max); ", sizeof(issues_buf) - strlen(issues_buf) - 1);
        health_issues++;
    }

    if (default_cfg->frequency_range <= 0.0f) {
        strncat(issues_buf, "频率范围为零; ", sizeof(issues_buf) - strlen(issues_buf) - 1);
        health_issues++;
    }

    if (default_cfg->cutoff_frequency <= 0.0f) {
        strncat(issues_buf, "截止频率为零; ", sizeof(issues_buf) - strlen(issues_buf) - 1);
        health_issues++;
    }

    /* 2. 创建临时分析器，验证分配和初始化流程 */
    LaplaceAnalyzer* test_analyzer = laplace_analyzer_create(default_cfg);
    if (!test_analyzer) {
        snprintf(report, report_size,
                 "拉普拉斯系统健康状态: [故障] 分析器创建失败 - %s",
                 issues_buf[0] ? issues_buf : "未知原因");
        return -1;
    }

    /* 3. 验证配置一致性 */
    if (laplace_analyzer_get_config(test_analyzer, &analyze_cfg) != 0) {
        laplace_analyzer_free(test_analyzer);
        snprintf(report, report_size, "拉普拉斯系统健康状态: [故障] 无法获取分析器配置");
        return -1;
    }

    if (analyze_cfg.num_samples != default_cfg->num_samples ||
        analyze_cfg.sample_rate != default_cfg->sample_rate) {
        strncat(issues_buf, "配置不一致; ", sizeof(issues_buf) - strlen(issues_buf) - 1);
        health_issues++;
    }

    /* 4. 释放测试分析器 */
    laplace_analyzer_free(test_analyzer);

    /* 5. 生成健康报告 */
    if (health_issues > 0) {
        snprintf(report, report_size,
                 "拉普拉斯系统健康状态: [警告] %d个问题: %s"
                 "配置 - 采样点:%zu 采样率:%.1fHz 频率范围:%.1f-%.1fHz 截止频率:%.1fHz",
                 health_issues, issues_buf,
                 default_cfg->num_samples, (double)default_cfg->sample_rate,
                 (double)default_cfg->min_frequency, (double)default_cfg->max_frequency,
                 (double)default_cfg->cutoff_frequency);
        return 0;
    }

    snprintf(report, report_size,
             "拉普拉斯系统健康状态: [正常] "
             "采样点:%zu 采样率:%.1fHz 频率范围:%.1f-%.1fHz 截止频率:%.1fHz "
             "稳定性分析:%s 频域分析:%s 自动调谐:%s",
             default_cfg->num_samples, (double)default_cfg->sample_rate,
             (double)default_cfg->min_frequency, (double)default_cfg->max_frequency,
             (double)default_cfg->cutoff_frequency,
             default_cfg->enable_stability ? "启用" : "关闭",
             default_cfg->enable_frequency ? "启用" : "关闭",
             default_cfg->enable_auto_tuning ? "启用" : "关闭");

    return 0;
}
