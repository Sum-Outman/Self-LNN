#include "selflnn/agi/capability_switch.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include <string.h>
#include <stdio.h>

/* 前向声明：子系统访问器（来自selflnn.h） */
extern void* selflnn_get_self_cognition(void);
extern void* selflnn_get_metacognition(void);
extern void* selflnn_get_online_learner(void);
extern void* selflnn_get_evolution_engine(void);
extern void* selflnn_get_thread_pool(void);
extern void* selflnn_get_dialogue_processor(void);
extern void* selflnn_get_planning_system(void);
extern void* selflnn_get_lnn(void);

static const char* g_capability_names[CAP_COUNT] = {
    "自我认知",
    "自我决策",
    "自主执行",
    "自我学习",
    "自我演化进化",
    "模仿学习",
    "自我修正",
    "自我反思",
    "好奇心",
    "规划能力",
    "对话能力",
    "并发能力"
};

static const char* g_capability_descriptions[CAP_COUNT] = {
    "监控系统状态、评估能力、跟踪学习进展、自我反思和元认知",
    "基于多属性效用理论进行单目标/多目标/约束/序列/博弈论决策",
    "自主执行任务调度、多机器人协调和设备控制",
    "强化学习、自监督学习、迁移学习等主动学习能力",
    "通过遗传算法、神经进化和进化策略实现自我演化",
    "行为克隆、DAgger、逆强化学习、生成对抗模仿学习",
    "贝叶斯根因诊断、自适应修正强度、多阶段验证流水线",
    "深度反思链驱动的自我反思和认知偏差检测",
    "主动探索未知领域和知识缺口的内在驱动力",
    "BFS/DFS/HTN层次任务网络规划",
    "基于LNN状态空间的自然语言对话生成",
    "多线程并行执行认知周期和并发任务调度"
};

static int g_capability_states[CAP_COUNT] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

/* 运行时强制锁定：被锁定后无法通过开关关闭 */
static int g_capability_forced_on[CAP_COUNT] = {0};

static CapabilityCheckFunc g_check_funcs[CAP_COUNT] = {NULL};
static CapabilitySetFunc g_set_funcs[CAP_COUNT] = {NULL};

static const int g_default_states[CAP_COUNT] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

/* ========== 真实回调函数实现 ========== */

/* F-025/F-036修复: 子系统启用/禁用API声明 — 通过selflnn API获取句柄后调用
 * 所有声明已通过 selflnn_get_*() 安全获取后再调用对应接口 */
extern int online_learner_set_enabled(void* learner, int enabled);
extern int evolution_engine_set_enabled(void* engine, int enabled);
extern int planning_system_set_enabled(void* planning, int enabled);
extern int dialogue_processor_set_enabled(void* processor, int enabled);
extern int self_cognition_set_enabled(void* sc, int enabled);
extern int metacognition_set_enabled(void* mc, int enabled);
extern int thread_pool_set_enabled(void* tp, int enabled);

/* 自我认知：检查自我认知系统是否创建 */
static int cap_check_self_cognition(void) {
    void* sc = selflnn_get_self_cognition();
    void* mc = selflnn_get_metacognition();
    return (sc && mc) ? 1 : 0;
}
static int cap_set_self_cognition(int enable) {
    g_capability_states[CAP_SELF_COGNITION] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_SELF_COGNITION]) { g_capability_states[CAP_SELF_COGNITION] = 1; return 0; }
    /* F-025: 实际启停子系统 */
    void* sc = selflnn_get_self_cognition();
    void* mc = selflnn_get_metacognition();
    if (sc) self_cognition_set_enabled(sc, enable ? 1 : 0);
    if (mc) metacognition_set_enabled(mc, enable ? 1 : 0);
    log_info("[能力开关] 自我认知 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 自我决策：检查决策引擎是否可用 */
static int cap_check_self_decision(void) {
    void* lnn = selflnn_get_lnn();
    return lnn ? 1 : 0;
}
static int cap_set_self_decision(int enable) {
    g_capability_states[CAP_SELF_DECISION] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_SELF_DECISION]) { g_capability_states[CAP_SELF_DECISION] = 1; return 0; }
    /* F-025: 自我决策影响在线学习器 */
    void* ol = selflnn_get_online_learner();
    if (ol) online_learner_set_enabled(ol, enable ? 1 : 0);
    log_info("[能力开关] 自我决策 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 自主执行：检查线程池是否可用 */
static int cap_check_autonomous_execution(void) {
    void* tp = selflnn_get_thread_pool();
    return tp ? 1 : 0;
}
static int cap_set_autonomous_execution(int enable) {
    g_capability_states[CAP_AUTONOMOUS_EXECUTION] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_AUTONOMOUS_EXECUTION]) { g_capability_states[CAP_AUTONOMOUS_EXECUTION] = 1; return 0; }
    /* F-025: 启停线程池 */
    void* tp = selflnn_get_thread_pool();
    if (tp) thread_pool_set_enabled(tp, enable ? 1 : 0);
    log_info("[能力开关] 自主执行 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 自我学习：检查在线学习器是否创建 */
static int cap_check_self_learning(void) {
    void* ol = selflnn_get_online_learner();
    return ol ? 1 : 0;
}
static int cap_set_self_learning(int enable) {
    g_capability_states[CAP_SELF_LEARNING] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_SELF_LEARNING]) { g_capability_states[CAP_SELF_LEARNING] = 1; return 0; }
    /* F-025: 启停在线学习器 */
    void* ol = selflnn_get_online_learner();
    if (ol) online_learner_set_enabled(ol, enable ? 1 : 0);
    log_info("[能力开关] 自我学习 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 自我演化：检查演化引擎是否创建 */
static int cap_check_self_evolution(void) {
    void* evo = selflnn_get_evolution_engine();
    return evo ? 1 : 0;
}
static int cap_set_self_evolution(int enable) {
    g_capability_states[CAP_SELF_EVOLUTION] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_SELF_EVOLUTION]) { g_capability_states[CAP_SELF_EVOLUTION] = 1; return 0; }
    /* F-025: 启停演化引擎 */
    void* evo = selflnn_get_evolution_engine();
    if (evo) evolution_engine_set_enabled(evo, enable ? 1 : 0);
    log_info("[能力开关] 自我演化 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 模仿学习：检查LNN训练能力 */
static int cap_check_imitation(void) {
    void* lnn = selflnn_get_lnn();
    return lnn ? 1 : 0;
}
static int cap_set_imitation(int enable) {
    g_capability_states[CAP_IMITATION_LEARNING] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_IMITATION_LEARNING]) { g_capability_states[CAP_IMITATION_LEARNING] = 1; return 0; }
    log_info("[能力开关] 模仿学习 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 自我修正：检查认知系统和LNN */
static int cap_check_self_correction(void) {
    void* sc = selflnn_get_self_cognition();
    void* lnn = selflnn_get_lnn();
    return (sc && lnn) ? 1 : 0;
}
static int cap_set_self_correction(int enable) {
    g_capability_states[CAP_SELF_CORRECTION] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_SELF_CORRECTION]) { g_capability_states[CAP_SELF_CORRECTION] = 1; return 0; }
    log_info("[能力开关] 自我修正 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 自我反思：检查元认知系统 */
static int cap_check_reflection(void) {
    void* mc = selflnn_get_metacognition();
    return mc ? 1 : 0;
}
static int cap_set_reflection(int enable) {
    g_capability_states[CAP_REFLECTION] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_REFLECTION]) { g_capability_states[CAP_REFLECTION] = 1; return 0; }
    /* F-025: 启停元认知系统 */
    void* mc = selflnn_get_metacognition();
    if (mc) metacognition_set_enabled(mc, enable ? 1 : 0);
    log_info("[能力开关] 自我反思 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 好奇心：检查LNN和探索能力 */
static int cap_check_curiosity(void) {
    void* lnn = selflnn_get_lnn();
    return lnn ? 1 : 0;
}
static int cap_set_curiosity(int enable) {
    g_capability_states[CAP_CURIOSITY] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_CURIOSITY]) { g_capability_states[CAP_CURIOSITY] = 1; return 0; }
    log_info("[能力开关] 好奇心 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 规划能力：检查规划系统 */
static int cap_check_planning(void) {
    void* ps = selflnn_get_planning_system();
    return ps ? 1 : 0;
}
static int cap_set_planning(int enable) {
    g_capability_states[CAP_PLANNING] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_PLANNING]) { g_capability_states[CAP_PLANNING] = 1; return 0; }
    /* F-025: 启停规划系统 */
    void* ps = selflnn_get_planning_system();
    if (ps) planning_system_set_enabled(ps, enable ? 1 : 0);
    log_info("[能力开关] 规划能力 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 对话能力：检查对话处理器 */
static int cap_check_dialogue(void) {
    void* dp = selflnn_get_dialogue_processor();
    return dp ? 1 : 0;
}
static int cap_set_dialogue(int enable) {
    g_capability_states[CAP_DIALOGUE] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_DIALOGUE]) { g_capability_states[CAP_DIALOGUE] = 1; return 0; }
    /* F-025: 启停对话处理器 */
    void* dp = selflnn_get_dialogue_processor();
    if (dp) dialogue_processor_set_enabled(dp, enable ? 1 : 0);
    log_info("[能力开关] 对话能力 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 并发能力：检查线程池 */
static int cap_check_concurrency(void) {
    void* tp = selflnn_get_thread_pool();
    return tp ? 1 : 0;
}
static int cap_set_concurrency(int enable) {
    g_capability_states[CAP_CONCURRENCY] = (enable != 0) ? 1 : 0;
    if (g_capability_forced_on[CAP_CONCURRENCY]) { g_capability_states[CAP_CONCURRENCY] = 1; return 0; }
    /* F-025: 启停线程池 */
    void* tp = selflnn_get_thread_pool();
    if (tp) thread_pool_set_enabled(tp, enable ? 1 : 0);
    log_info("[能力开关] 并发能力 %s", enable ? "开启" : "关闭");
    return 0;
}

/* 初始化时自动注册所有回调 */
static int g_callbacks_registered = 0;
static void ensure_callbacks_registered(void) {
    if (g_callbacks_registered) return;
    capability_register_module(CAP_SELF_COGNITION,       cap_check_self_cognition,    cap_set_self_cognition);
    capability_register_module(CAP_SELF_DECISION,         cap_check_self_decision,     cap_set_self_decision);
    capability_register_module(CAP_AUTONOMOUS_EXECUTION,  cap_check_autonomous_execution, cap_set_autonomous_execution);
    capability_register_module(CAP_SELF_LEARNING,         cap_check_self_learning,     cap_set_self_learning);
    capability_register_module(CAP_SELF_EVOLUTION,        cap_check_self_evolution,    cap_set_self_evolution);
    capability_register_module(CAP_IMITATION_LEARNING,    cap_check_imitation,         cap_set_imitation);
    capability_register_module(CAP_SELF_CORRECTION,       cap_check_self_correction,   cap_set_self_correction);
    capability_register_module(CAP_REFLECTION,            cap_check_reflection,        cap_set_reflection);
    capability_register_module(CAP_CURIOSITY,             cap_check_curiosity,         cap_set_curiosity);
    capability_register_module(CAP_PLANNING,              cap_check_planning,          cap_set_planning);
    capability_register_module(CAP_DIALOGUE,              cap_check_dialogue,          cap_set_dialogue);
    capability_register_module(CAP_CONCURRENCY,           cap_check_concurrency,       cap_set_concurrency);
    g_callbacks_registered = 1;
    log_info("[能力开关] 全部12个能力开关回调注册完成");
}

const char* capability_get_name(CapabilityType type)
{
    if (type < 0 || type >= CAP_COUNT) return NULL;
    return g_capability_names[type];
}

const char* capability_get_description(CapabilityType type)
{
    if (type < 0 || type >= CAP_COUNT) return NULL;
    return g_capability_descriptions[type];
}

int capability_is_enabled(CapabilityType type)
{
    if (type < 0 || type >= CAP_COUNT) return 0;
    ensure_callbacks_registered();
    if (g_check_funcs[type]) return g_check_funcs[type]();
    return g_capability_states[type];
}

int capability_set_enabled(CapabilityType type, int enabled)
{
    if (type < 0 || type >= CAP_COUNT) return -1;
    ensure_callbacks_registered();
    int val = (enabled != 0) ? 1 : 0;
    if (!val && g_capability_forced_on[type]) {
        log_warning("[能力开关] %s 被强制锁定，无法关闭", g_capability_names[type]);
        return -1;
    }
    g_capability_states[type] = val;
    if (g_set_funcs[type]) {
        g_set_funcs[type](val);
    }
    return 0;
}

int capability_force_on(CapabilityType type)
{
    if (type < 0 || type >= CAP_COUNT) return -1;
    g_capability_forced_on[type] = 1;
    g_capability_states[type] = 1;
    if (g_set_funcs[type]) g_set_funcs[type](1);
    log_info("[能力开关] %s 已强制锁定为开启", g_capability_names[type]);
    return 0;
}

int capability_unforce(CapabilityType type)
{
    if (type < 0 || type >= CAP_COUNT) return -1;
    g_capability_forced_on[type] = 0;
    return 0;
}

int capability_enable(CapabilityType type)
{
    return capability_set_enabled(type, 1);
}

int capability_disable(CapabilityType type)
{
    return capability_set_enabled(type, 0);
}

int capability_toggle(CapabilityType type)
{
    if (type < 0 || type >= CAP_COUNT) return -1;
    int current = g_capability_states[type];
    return capability_set_enabled(type, !current);
}

int capability_enable_all(void)
{
    int i;
    for (i = 0; i < CAP_COUNT; i++) {
        g_capability_states[i] = 1;
        if (g_set_funcs[i]) {
            g_set_funcs[i](1);
        }
    }
    return 0;
}

int capability_disable_all(void)
{
    int i;
    for (i = 0; i < CAP_COUNT; i++) {
        g_capability_states[i] = 0;
        if (g_set_funcs[i]) {
            g_set_funcs[i](0);
        }
    }
    return 0;
}

int capability_get_states(int* states, size_t max_count)
{
    if (!states) return -1;
    size_t i;
    size_t count = (max_count < (size_t)CAP_COUNT) ? max_count : (size_t)CAP_COUNT;
    for (i = 0; i < count; i++) {
        states[i] = g_capability_states[i];
    }
    return (int)count;
}

int capability_set_states(const int* states, size_t count)
{
    if (!states) return -1;
    size_t i;
    size_t n = (count < (size_t)CAP_COUNT) ? count : (size_t)CAP_COUNT;
    for (i = 0; i < n; i++) {
        g_capability_states[i] = (states[i] != 0) ? 1 : 0;
        if (g_set_funcs[i]) {
            g_set_funcs[i](g_capability_states[i]);
        }
    }
    return (int)n;
}

int capability_get_enabled_count(void)
{
    int count = 0;
    int i;
    for (i = 0; i < CAP_COUNT; i++) {
        if (g_capability_states[i]) count++;
    }
    return count;
}

int capability_reset_to_defaults(void)
{
    int i;
    for (i = 0; i < CAP_COUNT; i++) {
        g_capability_states[i] = g_default_states[i];
        if (g_set_funcs[i]) {
            g_set_funcs[i](g_default_states[i]);
        }
    }
    return 0;
}

int capability_register_module(CapabilityType type,
                                CapabilityCheckFunc check_func,
                                CapabilitySetFunc set_func)
{
    if (type < 0 || type >= CAP_COUNT) return -1;
    g_check_funcs[type] = check_func;
    g_set_funcs[type] = set_func;
    return 0;
}

/* ================================================================
 * K-023: 能力开关完整状态诊断
 * ================================================================ */

int capability_diagnose_all(char* json_buffer, size_t buffer_size)
{
    if (!json_buffer || buffer_size < 256) return -1;

    ensure_callbacks_registered();

    int pos = 0;
    pos += snprintf(json_buffer + pos, buffer_size - pos,
                    "{\"capabilities\":[");

    int first = 1;
    for (int i = 0; i < CAP_COUNT; i++) {
        int enabled = g_capability_states[i];
        int forced = g_capability_forced_on[i];

        /* 执行运行时健康检测 */
        int subsystem_healthy = enabled;
        if (g_check_funcs[i]) {
            subsystem_healthy = g_check_funcs[i]();
        }

        if (buffer_size - pos < 256) break;

        pos += snprintf(json_buffer + pos, buffer_size - pos,
                    "%s{\"id\":%d,\"name\":\"%s\",\"enabled\":%d,"
                    "\"forced\":%d,\"healthy\":%d,\"desc\":\"%s\"}",
                    first ? "" : ",",
                    i, g_capability_names[i], enabled,
                    forced, subsystem_healthy, g_capability_descriptions[i]);
        first = 0;
    }

    pos += snprintf(json_buffer + pos, buffer_size - pos,
                    "],\"total\":%d,\"enabled_count\":%d}",
                    CAP_COUNT, capability_get_enabled_count());

    /* 诊断日志 */
    log_info("[能力诊断] 12项能力: 启用=%d/12, 健康: 见JSON输出",
             capability_get_enabled_count());

    return pos;
}

int capability_health_check(void)
{
    ensure_callbacks_registered();

    int healthy_count = 0;
    int details[CAP_COUNT] = {0};

    for (int i = 0; i < CAP_COUNT; i++) {
        int health = 0;
        if (g_capability_states[i]) {
            if (g_check_funcs[i]) {
                health = g_check_funcs[i]();
            } else {
                health = 1;
            }
        }
        details[i] = health;
        if (health) healthy_count++;
    }

    log_info("[能力健康] %d/12项能力底层子系统可用", healthy_count);

    /* 对不健康的输出警告 */
    for (int i = 0; i < CAP_COUNT; i++) {
        if (g_capability_states[i] && !details[i]) {
            log_warning("[能力健康] %s: 开关已启用但底层子系统不可用",
                       g_capability_names[i]);
        }
    }

    return healthy_count;
}

/* ================================================================
 * P2-8: 能力依赖图与冲突检测
 * ================================================================ */

/* 能力依赖关系矩阵 [被依赖方][依赖方] = 1 表示依赖 */
/* 读取方式：cap_deps[前置能力][目标能力] = 1 表示"启用目标必须先启用前置" */
static const int g_capability_deps[CAP_COUNT][CAP_COUNT] = {
    /* 自我认知 */ {0,0,0,0,0,0,0,0,0,0,0,0},
    /* 自我决策 */ {0,0,1,1,0,0,0,0,0,1,0,0},  /* 自主执行、自我学习、规划依赖自我决策 */
    /* 自主执行 */ {0,0,0,0,0,0,0,0,0,0,0,0},
    /* 自我学习 */ {0,0,0,0,1,1,0,0,0,0,0,0},  /* 演化、模仿学习依赖自我学习 */
    /* 自我演化 */ {0,0,0,0,0,0,0,0,0,0,0,0},
    /* 模仿学习 */ {0,0,0,0,0,0,0,0,0,0,0,0},
    /* 自我修正 */ {0,0,0,0,0,0,0,0,0,0,0,0},
    /* 自我反思 */ {0,1,0,0,0,0,1,0,0,0,0,0},  /* 反思驱动决策和修正 */
    /* 好奇心 */   {0,0,0,0,0,0,0,0,0,0,0,0},
    /* 规划能力 */ {0,0,1,0,0,0,0,0,0,0,0,0},  /* 自主执行依赖规划 */
    /* 对话能力 */ {0,0,0,0,0,0,0,0,0,0,0,0},
    /* 并发能力 */ {0,0,0,0,0,0,0,0,0,0,0,0},
};

/* 能力冲突关系矩阵 [A][B] = 1 表示 A和B 不能同时启用 */
static const int g_capability_conflicts[CAP_COUNT][CAP_COUNT] = {0};

int capability_get_dependencies(CapabilityType type, int* dependencies, int max_count)
{
    if (type < 0 || type >= CAP_COUNT || !dependencies || max_count <= 0) return -1;

    int count = 0;
    for (int i = 0; i < CAP_COUNT && count < max_count; i++) {
        if (g_capability_deps[i][type]) {
            dependencies[count++] = i;
        }
    }
    return count;
}

int capability_check_conflict(CapabilityType type_a, CapabilityType type_b)
{
    if (type_a < 0 || type_a >= CAP_COUNT || type_b < 0 || type_b >= CAP_COUNT)
        return -1;
    return g_capability_conflicts[type_a][type_b];
}

int capability_enable_safe(CapabilityType type)
{
    if (type < 0 || type >= CAP_COUNT) return -1;

    /* 1. 检查依赖：必须所有前置依赖都可用 */
    for (int dep = 0; dep < CAP_COUNT; dep++) {
        if (g_capability_deps[dep][type]) {
            if (!g_capability_states[dep]) {
                log_warning("[能力开关] %s 依赖 %s 但 %s 未启用 — 自动启用",
                           g_capability_names[type],
                           g_capability_names[dep],
                           g_capability_names[dep]);
                /* 递归启用前置依赖 */
                if (capability_enable_safe((CapabilityType)dep) != 0) {
                    log_error("[能力开关] 无法启用依赖 %s", g_capability_names[dep]);
                    return -1;
                }
            }
        }
    }

    /* 2. 检查冲突 */
    for (int i = 0; i < CAP_COUNT; i++) {
        if (g_capability_conflicts[type][i] && g_capability_states[i]) {
            log_error("[能力开关] %s 与已启用的 %s 存在冲突",
                     g_capability_names[type], g_capability_names[i]);
            return -1;
        }
    }

    /* 3. 启用目标能力 */
    g_capability_states[type] = 1;
    if (g_set_funcs[type]) {
        g_set_funcs[type](1);
    }
    log_info("[能力开关] 安全启用 %s", g_capability_names[type]);
    return 0;
}

int capability_disable_safe(CapabilityType type)
{
    if (type < 0 || type >= CAP_COUNT) return -1;

    /* 检查是否有其他能力依赖它 */
    for (int i = 0; i < CAP_COUNT; i++) {
        if (g_capability_deps[type][i] && g_capability_states[i]) {
            log_error("[能力开关] %s 正在被 %s 依赖，无法禁用",
                     g_capability_names[type], g_capability_names[i]);
            return -1;
        }
    }

    /* 检查强制锁定 */
    if (g_capability_forced_on[type]) {
        log_warning("[能力开关] %s 已被强制锁定，无法禁用", g_capability_names[type]);
        return -1;
    }

    g_capability_states[type] = 0;
    if (g_set_funcs[type]) {
        g_set_funcs[type](0);
    }
    log_info("[能力开关] 安全禁用 %s", g_capability_names[type]);
    return 0;
}
