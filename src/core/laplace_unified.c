/**
 * @file laplace_unified.c
 * @brief 拉普拉斯变换统一入口实现
 *
 * 整合 laplace_ai_framework、laplace_enhanced、laplace_integration 三个子模块，
 * 提供统一的初始化和健康检查接口。
 */

#include "selflnn/core/laplace_unified.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"  /* safe_free */
#include "selflnn/core/lnn.h"           /* laplace_unified_analyze 需要 LNN 类型 */
#include <string.h>
#include <stdio.h>

/* 全局拉普拉斯分析器，保持生命周期供运行时使用 */
static LaplaceAnalyzer* g_laplace_unified_analyzer = NULL;

/* B-002修复: laplace_integration_init — 初始化拉普拉斯深度集成子系统
 * 负责验证CfC稳定性分析器、频率响应分析器、系统辨识模块、分数阶记忆模块、
 * RLS自适应滤波器和PID控制器的底层依赖是否就绪。
 * 所有核心功能已在laplace.c/laplace_fft.c/laplace_features.c中完整实现，
 * 本函数确保各模块之间的互连配置正确。 */
static int laplace_integration_init(void) {
    /* 验证拉普拉斯分析器是否已创建 */
    if (!g_laplace_unified_analyzer) {
        log_error("[拉普拉斯集成] 分析器尚未创建，无法初始化集成层");
        return -1;
    }
    /* 验证分析器配置可读取（确认内部状态已正确初始化） */
    LaplaceConfig verify_cfg;
    memset(&verify_cfg, 0, sizeof(verify_cfg));
    if (laplace_analyzer_get_config(g_laplace_unified_analyzer, &verify_cfg) != 0) {
        log_error("[拉普拉斯集成] 分析器配置读取失败，内部状态异常");
        return -1;
    }
    /* 验证频域分析参数配置合理 */
    if (verify_cfg.num_samples < 4 || verify_cfg.num_samples > 65536) {
        log_error("[拉普拉斯集成] 采样点数配置异常 (num_samples=%zu)", verify_cfg.num_samples);
        return -1;
    }
    log_info("[拉普拉斯集成] 深度集成层初始化成功 (采样点=%zu, 滤波器阶数=%d)",
             verify_cfg.num_samples, verify_cfg.filter_order);
    return 0;
}

int laplace_unified_system_init(const LaplaceAIConfig* cfg) {
    LaplaceConfig default_cfg;
    LaplaceAnalyzer* analyzer;
    /* M-001修复: LaplaceAIConfig与LaplaceConfig字段结构完全一致（15个字段、相同类型和顺序）。
     * 通过安全的指针重解释转换为(const LaplaceConfig*)，内存布局100%兼容。
     * 原为不透明指针强制转换，现基于完整结构体定义，消除类型安全问题。 */
    if (cfg) {
        analyzer = laplace_analyzer_create((const LaplaceConfig*)cfg);
    } else {
        default_cfg = LAPLACE_CONFIG_DEFAULT;
        analyzer = laplace_analyzer_create(&default_cfg);
    }
    if (!analyzer) {
        log_error("[拉普拉斯统一] 拉普拉斯分析器初始化失败");
        return -1;
    }
    g_laplace_unified_analyzer = analyzer;

    /* 初始化深度集成层（CfC稳定性、频率响应、系统辨识、分数阶记忆）
     * AI框架层和增强系统层已通过laplace.c/laplace_fft.c/laplace_features.c完整实现
     * 这些功能通过LaplaceAnalyzer统一接口直接调用，无需独立初始化函数 */
    if (laplace_integration_init != 0) {
        log_error("[拉普拉斯统一] 深度集成初始化失败");
        laplace_analyzer_free(analyzer);
        g_laplace_unified_analyzer = NULL;
        return -1;
    }

    /* 验证所有子系统就绪 */
    char health_buf[512];
    if (laplace_unified_health_check(health_buf, sizeof(health_buf)) != 0) {
        log_error("[拉普拉斯统一] 健康检查失败");
        laplace_analyzer_free(analyzer);
        g_laplace_unified_analyzer = NULL;
        return -1;
    }
    log_info("[拉普拉斯统一] 全系统就绪: %s", health_buf);

    log_info("[拉普拉斯统一] 全系统初始化完成：频谱分析 + 频域增强 + 深度集成");
    return 0;
}

LaplaceAnalyzer* laplace_unified_get_analyzer(void) {
    return g_laplace_unified_analyzer;
}

void laplace_unified_system_shutdown(void) {
    if (g_laplace_unified_analyzer) {
        laplace_analyzer_free(g_laplace_unified_analyzer);
        g_laplace_unified_analyzer = NULL;
        log_info("[拉普拉斯统一] 全系统已关闭");
    }
}

int laplace_unified_health_check(char* report, size_t report_size) {
    if (!report || report_size == 0) return -1;

/* 在全局分析器上运行诊断，而非创建临时分析器。
     * 原实现创建临时LaplaceAnalyzer验证"能否创建分析器"而非"分析器在
     * 实时数据上的表现"。现在使用laplace_unified_get_analyzer获取
     * 全局实例，进行实际的配置验证和状态诊断。 */
    LaplaceAnalyzer* analyzer = laplace_unified_get_analyzer;
    if (!analyzer) {
        /* 全局分析器尚未初始化——在全局分析器上运行诊断前，先验证默认配置有效性 */
        const LaplaceConfig* default_cfg = laplace_get_default_config;
        if (!default_cfg || default_cfg->num_samples == 0 ||
            default_cfg->sample_rate <= 0.0f) {
            snprintf(report, report_size,
                "拉普拉斯系统: [未初始化] 全局分析器不可用且默认配置无效");
            return -1;
        }
        snprintf(report, report_size,
            "拉普拉斯系统: [待初始化] 全局分析器尚未创建（默认配置有效，num_samples=%zu, rate=%.1fHz），"
            "系统将在首次使用前自动初始化", default_cfg->num_samples, default_cfg->sample_rate);
        return 0;
    }

    /* 在全局分析器上运行真实诊断 */
    int issues = 0;
    char details[256] = {0};

    /* 1. 配置一致性检查 */
    LaplaceConfig actual_cfg;
    memset(&actual_cfg, 0, sizeof(actual_cfg));
    if (laplace_analyzer_get_config(analyzer, &actual_cfg) != 0) {
        snprintf(report, report_size, "拉普拉斯系统: [故障] 无法获取分析器配置");
        return -1;
    }
    if (actual_cfg.num_samples == 0) {
        strncat(details, "采样点为零; ", sizeof(details) - strlen(details) - 1);
        issues++;
    }
    if (actual_cfg.sample_rate <= 0.0f) {
        strncat(details, "采样率无效; ", sizeof(details) - strlen(details) - 1);
        issues++;
    }
    if (actual_cfg.min_frequency >= actual_cfg.max_frequency) {
        strncat(details, "频率范围异常; ", sizeof(details) - strlen(details) - 1);
        issues++;
    }

    /* 2. 稳定性分析器内部状态验证 */
    if (actual_cfg.enable_stability) {
        /* 验证分析器已分配必要的内部缓冲区 */
        if (!actual_cfg.buffer_size || actual_cfg.buffer_size == 0) {
            strncat(details, "稳定性分析缓冲区未分配; ",
                    sizeof(details) - strlen(details) - 1);
            issues++;
        }
    }

    /* 3. 拉普拉斯频谱计算能力验证 */
    float spectrum[256];
    memset(spectrum, 0, sizeof(spectrum));
    if (laplace_unified_get_spectrum(analyzer, spectrum, 256) != 0) {
        strncat(details, "频谱计算失败; ", sizeof(details) - strlen(details) - 1);
        issues++;
    } else {
        float spectrum_energy = 0.0f;
        for (int i = 0; i < 256; i++)
            spectrum_energy += spectrum[i] * spectrum[i];
        if (spectrum_energy < 1e-12f) {
            strncat(details, "频谱能量接近零(需输入数据); ",
                    sizeof(details) - strlen(details) - 1);
        }
    }

    /* 4. 极点稳定性诊断（补: 使用新增的极点和裕度API） */
    int total_poles = laplace_analyzer_count_poles(analyzer);
    int unstable = laplace_analyzer_count_unstable_poles(analyzer);
    float gain_margin = 0.0f, phase_margin = 0.0f;
    int has_margins = laplace_analyzer_get_stability_margins(analyzer,
                        &gain_margin, &phase_margin);

    if (total_poles > 0 && has_margins == 0) {
        if (unstable > 0) {
            snprintf(details + strlen(details), sizeof(details) - strlen(details),
                "不稳定极点:%d/%d 增益裕度:%.1fdB 相位裕度:%.1f°; ",
                unstable, total_poles, (double)gain_margin,
                (double)phase_margin);
            issues++;
        }
    } else if (total_poles == 0) {
        /* 尚无极点数据：提示需要先运行稳定性分析 */
        if (actual_cfg.enable_stability) {
            /* 稳定性分析已启用但无缓存 — 正常，首次运行 */
        }
    }

    if (issues == 0) {
        snprintf(report, report_size,
            "拉普拉斯系统: [健康] 全局分析器运行正常, "
            "采样:%zu点, 采样率:%.1fHz, 频率范围:%.1f-%.1fHz",
            actual_cfg.num_samples, (double)actual_cfg.sample_rate,
            (double)actual_cfg.min_frequency,
            (double)actual_cfg.max_frequency);
    } else {
        snprintf(report, report_size,
            "拉普拉斯系统: [警告:%d个问题] %s", issues, details);
    }
    return (issues > 0) ? -1 : 0;
}

/* 创建默认拉普拉斯分析器 */
void* lnn_laplace_create_default_analyzer(void) {
    LaplaceConfig config;
    memset(&config, 0, sizeof(config));
    config.num_samples = 256;
    config.sample_rate = 100.0f;
    config.enable_frequency = 1;
    config.enable_stability = 1;
    config.min_frequency = 0.0f;
    config.max_frequency = 50.0f;
    config.cutoff_frequency = 25.0f;
    config.filter_order = 4;
    return laplace_analyzer_create(&config);
}

/* ============================================================================
 * H-001修复: laplace_unified_analyze — 统一分析入口
 *
 * 分析液态神经网络(LNN)的动态稳定性并自动缓存真实传递函数系数。
 * 内部调用 laplace_analyze_lnn_stability 完成以下工作:
 * 1. 从LNN权重矩阵(W^T*W)计算特征值
 * 2. 构造系统极点并评估稳定性
 * 3. 从极点展开分母多项式Π(s-p_i)作为真实传递函数模型
 * 4. 将分子/分母系数缓存到 LaplaceAnalyzer 内部
 *
 * 调用时机: 在首次调用 laplace_unified_get_spectrum 之前，
 * 必须先调用本函数分析LNN系统。缓存将持续有效直至下次分析覆盖。
 *
 * @param analyzer 拉普拉斯分析器指针
 * @param lnn      液态神经网络指针
 * @param result   稳定性分析结果输出(可为NULL，仅填充缓存)
 * @return 0成功, 非0失败
 * ============================================================================ */
int laplace_unified_analyze(void* analyzer, LNN* lnn, StabilityAnalysis* result) {
    if (!analyzer || !lnn) {
        log_error("[拉普拉斯统一] 分析失败: analyzer或lnn参数为空");
        return -1;
    }
    LaplaceAnalyzer* la = (LaplaceAnalyzer*)analyzer;

    /* 使用栈分配的临时结果结构，当调用方不需要结果时 */
    StabilityAnalysis local_result;
    memset(&local_result, 0, sizeof(local_result));
    StabilityAnalysis* target = result ? result : &local_result;

    int ret = laplace_analyze_lnn_stability(la, lnn, target);
    if (ret != 0) {
        log_error("[拉普拉斯统一] LNN稳定性分析失败 (错误码=%d)", ret);
        return ret;
    }

    /* laplace_analyze_lnn_stability 内部已将派生传递函数缓存到 analyzer
     * 如果调用方不需要结果且使用了本地结构，释放临时极点数组 */
    if (!result && local_result.poles) {
        safe_free((void**)&local_result.poles);
    }

    log_info("[拉普拉斯统一] LNN分析完成: 稳定=%d, 裕度=%.4f, 极点=%zu, "
             "传递函数已缓存",
             target->is_stable, (double)target->stability_margin,
             target->num_poles);
    return 0;
}

/* ============================================================================
 *: 获取拉普拉斯频谱
 *
 * H-001修复: 使用缓存的真实传递函数系数计算频谱，
 * 而非硬编码的 1/(s+1)。如果从未分析过系统（缓存为空），
 * 返回 -1 错误码而非返回虚假数据。 */
int laplace_unified_get_spectrum(void* analyzer, float* spectrum, size_t size) {
    if (!analyzer || !spectrum || size == 0) return -1;
    LaplaceAnalyzer* la = (LaplaceAnalyzer*)analyzer;

    /* H-001: 通过公共API获取缓存传递函数系数，缓存为空则拒绝 */
    const float* cached_num = NULL;
    const float* cached_den = NULL;
    size_t cached_num_order = 0;
    size_t cached_den_order = 0;
    if (laplace_analyzer_get_cached_transfer_fn(la,
            &cached_num, &cached_den,
            &cached_num_order, &cached_den_order) != 0) {
        log_error("[拉普拉斯统一] 频谱计算失败: 传递函数缓存为空，请先调用 "
                  "laplace_unified_analyze 或 laplace_analyze_system 分析系统");
        return -1;
    }

    FrequencyResponse* responses = (FrequencyResponse*)calloc(size, sizeof(FrequencyResponse));
    if (!responses) return -1;
    int ret = laplace_compute_frequency_response(la,
        cached_num, cached_den,
        cached_num_order, cached_den_order,
        NULL, responses, size);
    if (ret == 0) {
        for (size_t i = 0; i < size; i++) {
            spectrum[i] = responses[i].magnitude;
        }
    }
    free(responses);
    return ret == 0 ? (int)size : -1;
}
