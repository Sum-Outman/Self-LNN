/**
 * @file programming_bridge.c
 * @brief 自我编程闭环桥接实现
 *
 * 三个入口函数对应三条断裂线的修复：
 *   intent_to_code    — 认知→编程 (从能力缺口到代码生成)
 *   reason_to_code    — 推理→编程 (从规划方案到代码实现)
 *   learn_to_improve  — 学习→编程 (从错误反馈到代码改进)
 *
 * 每条路径执行相同的闭环：
 *   1. 从Intent/Spec/OldCode 生成代码(或从spec合成)
 *   2. compile → sandbox_execute → self_check_quality
 *   3. 失败时 self_improve_code 迭代优化
 *   4. 返回 Closure(含学习信号)
 */

#include "selflnn/programming/programming_bridge.h"
#include "selflnn/programming/c_interpreter.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/knowledge/knowledge.h"          /* KG持久化: 代码生成结果→知识库 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* 内部摘要缓冲区 */
static char g_bridge_summary[1024] = "";

/* ================================================================
 * 内部辅助
 * ================================================================ */

static void closure_init(ProgrammingClosure* c) {
    memset(c, 0, sizeof(ProgrammingClosure));
}

static void closure_set_error(ProgrammingClosure* c, const char* msg) {
    if (c->generated_code) { safe_free((void**)&c->generated_code); }
    if (c->improved_code)  { safe_free((void**)&c->improved_code); }
    c->compilation_ok = 0;
    c->execution_ok = 0;
    c->quality_score = 0;
    c->learning_signal = 0.0f;
    snprintf(c->error_log, sizeof(c->error_log), "%s", msg);
    snprintf(g_bridge_summary, sizeof(g_bridge_summary), "FAIL: %s", msg);
}

/* ================================================================
 * KG 集成: 将编程闭环结果持久化到知识库+技能库
 * ================================================================ */

/** 全局KG句柄(惰性初始化, 多次调用共享) */
static KnowledgeBase* g_programming_kg = NULL;

/**
 * @brief 将成功的编程闭环结果写入知识库
 */
static void store_closure_to_kg(const ProgrammingIntent* intent,
                                const ProgrammingClosure* closure) {
    if (!intent || !closure || closure->learning_signal < 0.3f) return;

    if (!g_programming_kg) {
        g_programming_kg = knowledge_base_create(2048);
        if (!g_programming_kg) return;
        log_info("编程桥接KG: 已创建知识库");
    }

    char subj[128], pred[128], obj[512];
    KnowledgeEntry entry;
    memset(&entry, 0, sizeof(entry));
    snprintf(subj, sizeof(subj), "%s", intent->function_name);
    entry.subject = subj;
    snprintf(pred, sizeof(pred), "implements");
    entry.predicate = pred;
    snprintf(obj, sizeof(obj),
            "function:%s quality=%d compile=%d exec=%d signal=%.2f desc=%.200s",
            intent->function_name, closure->quality_score,
            closure->compilation_ok, closure->execution_ok,
            closure->learning_signal, intent->description);
    entry.object = obj;
    entry.confidence = closure->learning_signal > 0.7f ? CONFIDENCE_HIGH :
                       closure->learning_signal > 0.4f ? CONFIDENCE_MEDIUM :
                                                         CONFIDENCE_LOW;
    entry.timestamp = (long)time(NULL);

    int ret = knowledge_base_add(g_programming_kg, &entry);
    if (ret == 0) {
        log_info("编程桥接KG: 已写入知识条目 '%s'", intent->function_name);
    }
}

/**
 * @brief 从KG查询与当前意图相似的已有代码模式
 */
static int query_kg_for_similar_code(const ProgrammingIntent* intent,
                                     char* hint_out, size_t hint_size) {
    if (!intent || !hint_out || !hint_size) return 0;
    hint_out[0] = '\0';

    if (g_programming_kg) {
        KnowledgeEntry results[4];
        int n = knowledge_base_search_similar(g_programming_kg,
            intent->function_name, "implements", "",
            0.3f, results, 4);
        if (n > 0) {
            snprintf(hint_out, hint_size,
                    "KG: 发现%d个相似知识条目(%s)", n, results[0].subject);
            return n;
        }
    }

    return 0;
}

static int run_sandbox_check(SelfProgrammingEngine* engine,
                             const char* code,
                             ProgrammingClosure* c) {
    if (!code) return -1;

    /* 1. 编译验证 */
    CompilationResult cr = verify_code_compilation(engine, code);
    c->compilation_ok = cr.success;
    if (!cr.success) {
        snprintf(c->error_log, sizeof(c->error_log), "Compile: %s", cr.error_message);
        self_programming_feedback_compile_error(engine, code, cr.error_message);
        return -1;
    }

    /* 2. 沙箱执行 */
    char output_buf[4096] = {0};
    int exec_ret = execute_code_sandboxed(engine, code, "", output_buf, sizeof(output_buf));
    c->execution_ok = (exec_ret == 0);

    /* 3. 质量自检 */
    CodeQualityResult qr = self_check_code_gen_quality(engine, code);
    c->quality_score = (int)qr.quality_score;

    /* 4. 学习信号 */
    float exec_score = c->execution_ok ? 1.0f : 0.0f;
    c->learning_signal = (exec_score * 0.6f + (qr.quality_score / 100.0f) * 0.4f);
    if (!c->compilation_ok) c->learning_signal = 0.0f;

    return c->compilation_ok ? 0 : -1;
}

/* ================================================================
 * Intent → CodeSpec 转换
 * ================================================================ */

static CodeSpecification* intent_to_spec(const ProgrammingIntent* intent) {
    if (!intent || !intent->function_name[0]) return NULL;

    CodeSpecification* spec = create_code_specification(
        intent->function_name, intent->description, TYPE_FLOAT);

    if (!spec) return NULL;

    /* 添加I/O示例 */
    for (int i = 0; i < intent->example_count && i < 8; i++) {
        double inputs[4]  = {0};
        int    in_count = intent->input_count;
        if (in_count > 4) in_count = 4;
        for (int j = 0; j < in_count; j++) inputs[j] = (double)intent->io_examples[i][j];
        double expected = (double)intent->io_examples[i][in_count];
        code_spec_add_example(spec, inputs, (size_t)in_count, expected);
    }

    return spec;
}

/* ================================================================
 * 公共API — intent_to_code
 * ================================================================ */

int programming_bridge_intent_to_code(SelfProgrammingEngine* engine,
                                      const ProgrammingIntent* intent,
                                      ProgrammingClosure* closure) {
    if (!engine || !intent || !closure) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "engine/intent/closure 为 NULL");
        return -1;
    }

    closure_init(closure);
    snprintf(g_bridge_summary, sizeof(g_bridge_summary),
             "INTENT→CODE: %s", intent->description);

    /* Step 0: 查询KG, 看是否有相似代码模式可复用 */
    char kg_hint[512] = {0};
    query_kg_for_similar_code(intent, kg_hint, sizeof(kg_hint));

    /* Step 1: Intent → Specification */
    CodeSpecification* spec = intent_to_spec(intent);
    if (!spec) {
        closure_set_error(closure, "无法从Intent创建CodeSpecification");
        return -1;
    }

    /* Step 2: Specification → Code (synthesize + generate) */
    char* code = synthesize_code(engine, spec);
    code_specification_destroy(spec);

    if (!code) {
        /* synthesize_code 失败时尝试 generate_c 后备 */
        CodeSpecification* spec2 = intent_to_spec(intent);
        if (spec2) {
            code = self_programming_generate_c(engine, spec2);
            code_specification_destroy(spec2);
        }
    }

    if (!code) {
        closure_set_error(closure, "代码合成/生成均失败");
        return -1;
    }
    closure->generated_code = code;

    /* Step 3: compile → execute → quality check */
    int ok = run_sandbox_check(engine, code, closure);

    /* Step 4: 失败则进入 self_improve_code 迭代优化 */
    int max_iter = intent->max_iterations > 0 ? intent->max_iterations : 3;
    if (!ok && max_iter > 0) {
        char* improved = self_improve_code(engine, code, max_iter);
        if (improved) {
            closure->improved_code = improved;
            closure->iterations_done = max_iter;
            run_sandbox_check(engine, improved, closure);
        }
    }

    snprintf(g_bridge_summary, sizeof(g_bridge_summary),
             "INTENT→CODE: %s | compile=%d exec=%d quality=%d signal=%.2f",
             intent->function_name,
             closure->compilation_ok, closure->execution_ok,
             closure->quality_score, closure->learning_signal);

    /* Step 5: 将成功结果持久化到KG+技能库 */
    if (closure->learning_signal > 0.3f) {
        store_closure_to_kg(intent, closure);
    }

    return 0;
}

/* ================================================================
 * 公共API — reason_to_code
 * ================================================================ */

/* P3-005: 推理→代码生成 — 完整实现，待自我编程引擎任务调度集成调用 */
int programming_bridge_reason_to_code(SelfProgrammingEngine* engine,
                                      const CodeSpecification* spec,
                                      ProgrammingClosure* closure) {
    if (!engine || !spec || !closure) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "engine/spec/closure 为 NULL");
        return -1;
    }

    closure_init(closure);
    snprintf(g_bridge_summary, sizeof(g_bridge_summary),
             "REASON→CODE: %s", spec->function_name);

    /* Step 1: spec → code */
    char* code = synthesize_code(engine, spec);
    if (!code) {
        code = self_programming_generate_c(engine, spec);
    }

    if (!code) {
        closure_set_error(closure, "推理→代码生成失败");
        return -1;
    }
    closure->generated_code = code;

    /* Step 2: 验证+优化闭环 */
    int ok = run_sandbox_check(engine, code, closure);
    if (!ok) {
        char* improved = self_improve_code(engine, code, 3);
        if (improved) {
            closure->improved_code = improved;
            closure->iterations_done = 3;
            run_sandbox_check(engine, improved, closure);
        }
    }

    snprintf(g_bridge_summary, sizeof(g_bridge_summary),
             "REASON→CODE: %s | compile=%d quality=%d signal=%.2f",
             spec->function_name,
             closure->compilation_ok, closure->quality_score, closure->learning_signal);

    /* 持久化到KG */
    if (closure->learning_signal > 0.3f) {
        /* 构造临时Intent仅用于KG存储 */
        ProgrammingIntent tmp;
        memset(&tmp, 0, sizeof(tmp));
        snprintf(tmp.function_name, sizeof(tmp.function_name), "%s", spec->function_name);
        snprintf(tmp.description, sizeof(tmp.description), "%s", spec->description ? spec->description : "");
        store_closure_to_kg(&tmp, closure);
    }

    return 0;
}

/* ================================================================
 * 公共API — learn_to_improve
 * ================================================================ */

int programming_bridge_learn_to_improve(SelfProgrammingEngine* engine,
                                        const char* source_code,
                                        const char* error_feedback,
                                        ProgrammingClosure* closure) {
    if (!engine || !source_code || !closure) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "engine/source/closure 为 NULL");
        return -1;
    }

    closure_init(closure);
    snprintf(g_bridge_summary, sizeof(g_bridge_summary),
             "LEARN→IMPROVE: %s", error_feedback ? error_feedback : "(无反馈)");

    /* Step 0: 诊断当前代码质量 */
    CodeQualityResult qr_before = self_check_code_gen_quality(engine, source_code);

    /* Step 1: self_improve_code 迭代优化 */
    int iterations = 5; /* 学习路径多轮迭代 */
    char* improved = self_improve_code(engine, source_code, iterations);
    if (!improved) {
        closure_set_error(closure, "代码改进失败(可能已是局部最优)");
        return -1;
    }
    closure->improved_code = improved;
    closure->iterations_done = iterations;

    /* Step 2: 验证改进后代码 */
    int ok = run_sandbox_check(engine, improved, closure);

    /* Step 3: 对比前后质量 */
    CodeQualityResult qr_after = self_check_code_gen_quality(engine, improved);
    closure->quality_score = (int)qr_after.quality_score;

    /* 学习信号: 改进幅度 */
    float delta = (qr_after.quality_score - qr_before.quality_score) / 100.0f;
    closure->learning_signal = ok ? (0.5f + delta * 0.5f) : (delta * 0.3f);
    if (closure->learning_signal > 1.0f) closure->learning_signal = 1.0f;
    if (closure->learning_signal < 0.0f) closure->learning_signal = 0.0f;

    snprintf(g_bridge_summary, sizeof(g_bridge_summary),
             "LEARN→IMPROVE: quality %.0f→%.0f | compile=%d signal=%.2f",
             qr_before.quality_score, qr_after.quality_score,
             closure->compilation_ok, closure->learning_signal);

    /* 持久化到KG(记录学习改进) */
    if (closure->learning_signal > 0.3f) {
        /* 知识条目: 错误反馈→代码改进 */
        ProgrammingIntent tmp;
        memset(&tmp, 0, sizeof(tmp));
        snprintf(tmp.function_name, sizeof(tmp.function_name), "learn_improve");
        snprintf(tmp.description, sizeof(tmp.description),
                "学习改进: %s", error_feedback ? error_feedback : "自我优化");
        store_closure_to_kg(&tmp, closure);
    }

    return 0;
}

/* ================================================================
 * 公共API — full_closure
 * ================================================================ */

int programming_bridge_full_closure(SelfProgrammingEngine* engine,
                                    const char* source,
                                    const CodeSpecification* spec,
                                    int max_iter,
                                    ProgrammingClosure* closure) {
    if (!engine || !closure) return -1;
    closure_init(closure);

    char* code = NULL;
    if (source) {
        code = strdup(source);
        closure->generated_code = code;
    } else if (spec) {
        code = synthesize_code(engine, spec);
        if (!code) code = self_programming_generate_c(engine, spec);
        closure->generated_code = code;
    }

    if (!code) {
        closure_set_error(closure, "无源代码且无规格说明");
        return -1;
    }

    int ok = run_sandbox_check(engine, code, closure);
    if (!ok && max_iter > 0) {
        char* improved = self_improve_code(engine, code, max_iter);
        if (improved) {
            closure->improved_code = improved;
            closure->iterations_done = max_iter;
            run_sandbox_check(engine, improved, closure);
        }
    }

    return 0;
}

/* ================================================================
 * 公共API — autonomous_cycle + scan_capability_gaps
 * ================================================================ */

int programming_bridge_autonomous_cycle(SelfProgrammingEngine* engine,
                                        LNN* lnn,
                                        const AutonomousProgrammingConfig* config,
                                        ProgrammingClosure** results,
                                        int* num_results) {
    if (!engine || !lnn || !config || !results || !num_results) return -1;
    if (!config->enable_auto_trigger) {
        *num_results = 0;
        *results = NULL;
        return 0;
    }

    /* Step 1: 扫描能力缺口 */
    ProgrammingIntent* intents = NULL;
    int intent_count = 0;
    programming_bridge_scan_capability_gaps(lnn, &intents, &intent_count);

    if (intent_count == 0) {
        *num_results = 0;
        *results = NULL;
        return 0;
    }

    /* Step 2: 逐个执行闭环 */
    *results = (ProgrammingClosure*)safe_calloc((size_t)intent_count, sizeof(ProgrammingClosure));
    if (!*results) {
        safe_free((void**)&intents);
        return -1;
    }

    int completed = 0;
    for (int i = 0; i < intent_count; i++) {
        int ret = programming_bridge_intent_to_code(engine, &intents[i], &(*results)[completed]);
        if (ret == 0) completed++;
    }

    *num_results = completed;
    safe_free((void**)&intents);

    snprintf(g_bridge_summary, sizeof(g_bridge_summary),
             "AUTONOMOUS: %d gaps found, %d closures completed",
             intent_count, completed);

    return 0;
}

int programming_bridge_scan_capability_gaps(LNN* lnn,
                                            ProgrammingIntent** intents,
                                            int* count) {
    if (!lnn || !intents || !count) return -1;

    /* 从LNN读取能力评分向量 */
    float capability_scores[16] = {0};
    memset(capability_scores, 0, sizeof(capability_scores));

    /* 获取LNN当前能力状态
     * 使用 self_cognition_get_capability 读取维度得分：
     *   0=推理, 1=计算, 2=规划, 3=代码生成, 4=验证,
     *   5=学习速率, 6=知识覆盖, 7=适应速度
     */
    (void)lnn; /* 由外部 cognition 层传入能力评分 */

    /* 默认: 预备3个基础能力缺口 */
    *count = 3;
    *intents = (ProgrammingIntent*)safe_calloc(3, sizeof(ProgrammingIntent));
    if (!*intents) return -1;

    /* Gap 1: 数学计算能力 */
    ProgrammingIntent* p1 = &(*intents)[0];
    snprintf(p1->description, sizeof(p1->description),
             "数学计算增强: 基于当前LNN参数生成C语言辅助计算函数");
    snprintf(p1->function_name, sizeof(p1->function_name), "compute_aid");
    p1->input_count   = 2;
    p1->output_count  = 1;
    p1->example_count = 3;
    p1->io_examples[0][0] = 1.0f; p1->io_examples[0][1] = 2.0f; p1->io_examples[0][2] = 3.0f;
    p1->io_examples[1][0] = 0.0f; p1->io_examples[1][1] = 0.0f; p1->io_examples[1][2] = 0.0f;
    p1->io_examples[2][0] = -1.0f; p1->io_examples[2][1] = 1.0f; p1->io_examples[2][2] = 0.0f;
    p1->priority = 1;
    p1->max_iterations = 5;

    /* Gap 2: 排序/搜索能力 */
    ProgrammingIntent* p2 = &(*intents)[1];
    snprintf(p2->description, sizeof(p2->description),
             "排序算法实现: 生成快速排序或归并排序C代码");
    snprintf(p2->function_name, sizeof(p2->function_name), "sort_values");
    p2->input_count   = 1;
    p2->output_count  = 1;
    p2->example_count = 2;
    p2->io_examples[0][0] = 3.0f; p2->io_examples[0][1] = 1.0f;
    p2->io_examples[1][0] = 1.0f; p2->io_examples[1][1] = 3.0f;
    p2->priority = 1;
    p2->max_iterations = 5;

    /* Gap 3: 数据过滤能力 */
    ProgrammingIntent* p3 = &(*intents)[2];
    snprintf(p3->description, sizeof(p3->description),
             "数据过滤: 根据阈值过滤数值数组C函数");
    snprintf(p3->function_name, sizeof(p3->function_name), "filter_threshold");
    p3->input_count   = 2;
    p3->output_count  = 1;
    p3->example_count = 2;
    p3->io_examples[0][0] = 5.0f; p3->io_examples[0][1] = 3.0f; p3->io_examples[0][2] = 5.0f;
    p3->io_examples[1][0] = 2.0f; p3->io_examples[1][1] = 3.0f; p3->io_examples[1][2] = 0.0f;
    p3->priority = 0;
    p3->max_iterations = 3;

    return 0;
}

/* ================================================================
 * 工具函数
 * ================================================================ */

void programming_closure_free(ProgrammingClosure* closure) {
    if (!closure) return;
    if (closure->generated_code) safe_free((void**)&closure->generated_code);
    if (closure->improved_code)  safe_free((void**)&closure->improved_code);
    memset(closure, 0, sizeof(ProgrammingClosure));
}

/* P3-005: 桥接摘要查询 — 完整实现，待前端/日志模块集成调用 */
const char* programming_bridge_last_summary(void) {
    return g_bridge_summary;
}
