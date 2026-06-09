/**
 * @file programming_bridge.h
 * @brief 自我编程闭环桥接 — 连接认知·推理·学习与编程模块
 *
 * 现有系统拥有完整的代码分析/验证/优化工具链，但缺少三个关键连接：
 *   1. 认知(理解需求) → 编程(生成代码)        [认知→编程 断裂]
 *   2. 推理(规划方案) → 编程(实现方案)        [推理→编程 断裂]
 *   3. 学习(错误反馈) → 编程(改进代码)        [学习→编程 弱连接]
 *
 * 本模块将三条断裂线全部连接，形成完整的自我编程闭环：
 *   认知需求 → 推理规划 → 代码生成 → 编译验证 → 错误学习 → 自我改进 → 认知更新
 *
 * 三阶段补全方案：
 *   阶段1(本模块): 认知/推理/学习三个入口 → 编程模块委托
 *   阶段2(后续):   自主触发机制(认知自主决定何时写代码)
 *   阶段3(后续):   复杂代码生成(超越I/O示例推断, 使用LLM/MCTS/NAS)
 */

#ifndef SELFLNN_PROGRAMMING_BRIDGE_H
#define SELFLNN_PROGRAMMING_BRIDGE_H

#include "selflnn/core/lnn.h"
#include "selflnn/programming/self_programming.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * 类型定义
 * ================================================================ */

/**
 * @brief 编程意图——认知/推理模块触发代码生成时的需求描述
 *
 * 由认知模块(自我评估→发现能力缺口)或推理模块(规划→发现代码可解子目标)
 * 创建，传递给桥接层执行完整的代码生成→验证→改进闭环。
 */
typedef struct {
    char  description[1024];  /**< 自然语言需求描述 */
    char  function_name[128]; /**< 目标函数名 */
    int   input_count;        /**< 输入参数数量 */
    int   output_count;       /**< 输出数量 */
    float io_examples[8][4];  /**< I/O示例(最多8组, 每组[输入..., 输出]) */
    int   example_count;      /**< 实际示例数 */
    int   priority;           /**< 优先级 0=低 1=中 2=高 */
    int   max_iterations;     /**< 自我改进最大迭代轮数 */
} ProgrammingIntent;

/**
 * @brief 编程闭环结果——包含完整的生成/验证/改进链路输出
 */
typedef struct {
    char*  generated_code;    /**< 生成的源代码(malloc, 调用者释放) */
    char*  improved_code;     /**< 优化后的源代码(malloc, 调用者释放) */
    int    compilation_ok;    /**< 编译是否通过 */
    int    execution_ok;      /**< 沙箱执行是否通过 */
    int    quality_score;     /**< 代码质量评分 0-100 */
    int    iterations_done;   /**< 实际改进轮数 */
    char   error_log[4096];   /**< 编译/执行错误日志 */
    float  learning_signal;   /**< 学习信号 (0.0=失败 1.0=完美) */
} ProgrammingClosure;

/**
 * @brief 自主编程触发配置
 */
typedef struct {
    float capability_gap_threshold;    /**< 能力缺口阈值 (低于此值触发编程) */
    float improvement_signal_threshold; /**< 改进信号阈值 */
    int   max_autonomous_iterations;    /**< 最大自主迭代次数 */
    int   enable_auto_trigger;          /**< 是否启用自主触发 */
} AutonomousProgrammingConfig;

/* ================================================================
 * 核心桥接API — 三条断裂线的入口
 * ================================================================ */

/**
 * @brief 【认知→编程】认知评估发现能力缺口后，委托编程模块生成代码
 *
 * 调用路径: metacognition_neutral_self_assessment → 发现缺口
 *          → programming_bridge_intent_to_code → 整个闭环
 *
 * @param engine  编程引擎
 * @param intent  需求描述(I/O示例+函数签名)
 * @param closure 输出闭环结果(调用者用 programming_closure_free 释放)
 * @return 0=成功, -1=失败
 */
int programming_bridge_intent_to_code(SelfProgrammingEngine* engine,
                                      const ProgrammingIntent* intent,
                                      ProgrammingClosure* closure);

/**
 * @brief 【推理→编程】推理链拆解出代码可实现子目标后委托编程模块
 *
 * 调用路径: dtc_reason_chain → dtc_chain_to_plan → 发现代码可解步骤
 *          → programming_bridge_reason_to_code → 整个闭环
 *
 * @param engine  编程引擎
 * @param spec    代码规格说明(函数签名+前置/后置条件+I/O示例)
 * @param closure 输出闭环结果
 * @return 0=成功, -1=失败
 */
int programming_bridge_reason_to_code(SelfProgrammingEngine* engine,
                                      const CodeSpecification* spec,
                                      ProgrammingClosure* closure);

/**
 * @brief 【学习→编程】学习模块检测到代码缺陷后委托编程模块改进
 *
 * 调用路径: online_learning → 检测性能下降
 *          → meta_learning → 分析根因在代码
 *          → programming_bridge_learn_to_improve → 改进闭环
 *
 * @param engine      编程引擎
 * @param source_code  待改进的源代码
 * @param error_feedback 学习模块提供的错误反馈
 * @param closure      输出闭环结果
 * @return 0=成功, -1=失败
 */
int programming_bridge_learn_to_improve(SelfProgrammingEngine* engine,
                                        const char* source_code,
                                        const char* error_feedback,
                                        ProgrammingClosure* closure);

/* ================================================================
 * 自主触发API — 阶段2
 * ================================================================ */

/**
 * @brief 自主编程循环 — 认知系统自主决定何时写代码
 *
 * 调用路径: 认知周期性自我评估 → 检测能力缺口 → 创建Intent → 执行闭环
 *
 * @param engine    编程引擎
 * @param lnn       关联的LNN实例(用于评估当前能力)
 * @param config    自主配置
 * @param results   输出的所有完成项(动态数组, 调用者释放)
 * @param num_results 结果数量
 * @return 0=成功, -1=失败
 */
int programming_bridge_autonomous_cycle(SelfProgrammingEngine* engine,
                                        LNN* lnn,
                                        const AutonomousProgrammingConfig* config,
                                        ProgrammingClosure** results,
                                        int* num_results);

/**
 * @brief 认知能力缺口扫描 → 自动生成编程意图列表
 *
 * 扫描认知系统当前状态，发现需要编程补足的缺口，
 * 自动生成对应的 ProgrammingIntent 列表。
 *
 * @param lnn     LNN实例(读取能力评分)
 * @param intents 输出的意图列表(调用者 free)
 * @param count   意图数量
 * @return 0=成功, -1=失败
 */
int programming_bridge_scan_capability_gaps(LNN* lnn,
                                            ProgrammingIntent** intents,
                                            int* count);

/* ================================================================
 * 工具API
 * ================================================================ */

/**
 * @brief 执行完整的编程闭环(生成→编译→执行→改进)，直到通过或达到最大轮数
 *
 * @param engine  编程引擎
 * @param source  初始源代码(可为NULL, 则从spec生成)
 * @param spec    代码规格(可为NULL, 则仅优化source)
 * @param max_iter 最大迭代轮数
 * @param closure 输出
 * @return 0=成功, -1=失败
 */
int programming_bridge_full_closure(SelfProgrammingEngine* engine,
                                    const char* source,
                                    const CodeSpecification* spec,
                                    int max_iter,
                                    ProgrammingClosure* closure);

/**
 * @brief 释放闭环结果
 */
void programming_closure_free(ProgrammingClosure* closure);

/**
 * @brief 获取最后一次闭环的执行摘要
 */
const char* programming_bridge_last_summary(void);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_PROGRAMMING_BRIDGE_H */
