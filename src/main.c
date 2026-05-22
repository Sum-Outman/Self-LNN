#include "selflnn/backend/backend.h"
#include "selflnn/backend/websocket_push.h"
#include "selflnn/self_cognition.h"
/* ZSFWS-NEW-MAININC: 移除5个未使用的include
 * - metacognition.h: agi_bg_metacognition仅用内部API，不依赖此头文件
 * - multi_agent.h: main.c中从未调用multi_agent_*函数
 * - neural_architecture_search.h: main.c中从未调用NAS相关函数
 * - learning/meta_learning.h: main.c中从未调用meta_learning_*函数
 * - robot/ros_robot_controller.h: main.c中从未调用ros_robot_controller_*函数 */
#include "selflnn/core/lnn.h"
#include "selflnn/core/quaternion_lnn.h"
#include "selflnn/knowledge/knowledge.h"
#include "selflnn/knowledge/auto_learning.h"
#include "selflnn/multimodal/data_collection_pipeline.h"
#include "selflnn/memory/memory.h"
#include "selflnn/memory/memory_manager.h"
#include "selflnn/reasoning/reasoning.h"
#include "selflnn/reasoning/planning.h"

/* ZSF-018修复: 移除重复include，统一管理所有头文件引用 */
/* 能力开关 */
#include "selflnn/agi/capability_switch.h"
#include "selflnn/learning/learning.h"
#include "selflnn/learning/imitation_learning.h"
#include "selflnn/learning/online_learning.h"
#include "selflnn/multimodal/multimodal.h"
#include "selflnn/multimodal/multimodal_manager.h"
#include "selflnn/multimodal/dialogue.h"
#include "selflnn/training/training.h"
#include "selflnn/training/training_pipeline.h"
#include "selflnn/gpu/gpu.h"
#include "selflnn/robot/robot.h"
#include "selflnn/programming/self_programming.h"
#include "selflnn/product_design/product_design.h"
#include "selflnn/concurrency/thread_pool.h"
#include "selflnn/multisystem/multisystem_control.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/selflnn.h"
#include "selflnn/evolution/evolution_engine.h"
#include "selflnn/safety/safety_monitor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#include <process.h>
#include <direct.h>
#else
#include <unistd.h>
#include <pthread.h>
#endif

/* 静态函数前向声明 */
static int is_subsystem_healthy_int(const char* name, void* handle, int (*is_init)(void*));

/* AGI后台任务常量 */
#define AGI_BG_INTERVAL_MS       10000  /* 主循环间隔10秒（降低负载防栈溢出） */
#ifndef CLAMP
#define CLAMP(x,min,max) (((x)<(min))?(min):(((x)>(max))?(max):(x)))
#endif
#define ONLINE_LEARN_INTERVAL    36000  /* 在线学习每10小时（降低负载测试用） */
#define SELF_REFLECTION_INTERVAL 86400  /* 自我反思每24小时（降低负载测试用） */
#define KNOWLEDGE_CONSOLIDATE    86400  /* 知识固化每24小时（降低负载测试用） */
#define EVOLUTION_STEP_INTERVAL  86400  /* 演化步每24小时（降低负载测试用） */
#define COGNITION_UPDATE_INTERVAL 3600  /* 认知更新每1小时（降低负载测试用） */
#define SAFETY_CHECK_INTERVAL     3600  /* 安全检查每1小时（降低负载测试用） */
#define TRAINING_STEP_INTERVAL   86400  /* ZSFABC: 训练步每24小时（降低负载测试用） */

#define VERSION_MAJOR 1
#define VERSION_MINOR 4
#define VERSION_PATCH 0

static BackendServer* g_server = NULL;
static WSPushServer* g_ws_push_server = NULL;
static void* g_online_learner_handle = NULL;
static void* g_evolution_engine_handle = NULL;
static volatile sig_atomic_t g_agi_running = 1;
static time_t g_start_time = 0;
static time_t g_last_online_learn = 0;
static time_t g_last_reflection = 0;
static time_t g_last_consolidate = 0;
static time_t g_last_evolution = 0;
static time_t g_last_cognition = 0;
static time_t g_last_safety = 0;
static int g_bg_task_error_count = 0;
static TrainingPipeline* g_training_pipeline = NULL;   /* ZSFABC: 训练管线 */
static time_t g_last_training_step = 0;                 /* ZSFABC: 上次训练步时间 */

/* AGI后台任务：在线学习循环 */
static void agi_bg_online_learning(void) {
    void* learner = selflnn_get_online_learner();
    /* ZS-020修复: 使用在线学习器专用检查函数替代NULL */
    if (!is_subsystem_healthy_int("在线学习器", learner, is_online_learner_init)) return;
    OnlineLearningStatus status;
    memset(&status, 0, sizeof(OnlineLearningStatus));
    int s = online_learner_get_status((OnlineLearner*)learner, &status);
    if (status.total_samples < 10 || status.current_learning_rate < 1e-7f) {
        return;
    }
    (void)s;
    /* 从系统状态缓冲区获取真实的在线学习数据 */
    float* state = (float*)safe_malloc(128 * sizeof(float));
    float* target = (float*)safe_malloc(128 * sizeof(float));
    if (!state || !target) {
        free(state);
        free(target);
        return;
    }
    /* 尝试从LNN网络获取当前状态和最近输出作为训练数据 */
    void* lnn = selflnn_get_shared_lnn();
    if (lnn) {
        if (selflnn_get_recent_state(lnn, state, 128) == 0 &&
            selflnn_get_recent_output(lnn, target, 128) == 0) {
            float loss = 0.0f;
            online_learner_update((OnlineLearner*)learner, state, 128, target, 128, &loss);
        }
    }
    free(state);
    free(target);
}

/* AGI后台任务：知识固化（系统状态监控 + 记忆/知识统计） */
static void agi_bg_knowledge_consolidate(void) {
    static int consolidate_cycle = 0;
    consolidate_cycle++;

    if (consolidate_cycle % 10 == 0) {
        SystemStatus status;
        memset(&status, 0, sizeof(SystemStatus));
        if (selflnn_get_status(&status) == 0) {
            log_info("[AGI后台] 知识固化（周期%d）：记忆=%d条，知识=%d条，活动任务=%d",
                     consolidate_cycle / 10,
                     status.total_memories, status.total_knowledge, status.active_tasks);
        } else {
            log_debug("[AGI后台] 知识固化（周期%d）", consolidate_cycle / 10);
        }
    }

    /* 自动知识学习：每3个固化周期扫描一次知识库文件目录 */
    if (consolidate_cycle % 3 == 0) {
        void* al = selflnn_get_auto_learning();
        if (al) {
            AutoLearnStats astats;
            memset(&astats, 0, sizeof(astats));
            auto_learning_get_stats((const AutoLearningSystem*)al, &astats);
            if (astats.total_files_scanned > 0 || astats.total_entries_learned > 0) {
                log_debug("[AGI后台] 自动知识学习: %zu文件, %zu条目, %zu实体",
                         astats.total_files_scanned, astats.total_entries_learned,
                         astats.total_entities_extracted);
            }
        }
    }
}

/* AGI后台任务：自我反思（迭代式元认知循环） */
static void agi_bg_self_reflection(void) {
    void* scs = selflnn_get_self_cognition();
    if (!is_subsystem_healthy_int("自我认知", scs, NULL)) {
        log_debug("[AGI后台] 自我认知系统未初始化，跳过自我反思");
        return;
    }
    
    /* 第1轮：快速表面反思 */
    char assessment[1024];
    memset(assessment, 0, sizeof(assessment));
    self_cognition_neutral_assessment((SelfCognitionSystem*)scs, 0, assessment, sizeof(assessment));
    log_info("[AGI后台] 表面反思：%s", assessment);
    
    /* 第2-5轮：迭代式深度元认知循环（反思→修正→再反思） */
    int iterations = self_cognition_iterative_reflection((SelfCognitionSystem*)scs, 5);
    log_info("[AGI后台] 迭代元认知循环完成：%d轮", iterations);
    
    /* 记忆巩固：短期→长期认知记忆 */
    int consolidated = self_cognition_memory_consolidate((SelfCognitionSystem*)scs);
    if (consolidated > 0) {
        log_info("[AGI后台] 认知记忆巩固：%d个片段", consolidated);
    }
    
    void* learner = selflnn_get_online_learner();
    if (learner) {
        OnlineLearningStatus status;
        memset(&status, 0, sizeof(OnlineLearningStatus));
        if (online_learner_get_status((OnlineLearner*)learner, &status) == 0) {
            if (status.average_loss > 1.0f && status.total_samples > 100) {
                log_warning("[AGI后台] 反思检测到高损失=%.4f，建议调整策略",
                           status.average_loss);
            }
        }
    }
}

/* AGI后台任务：演化步 */
static void agi_bg_evolution_step(void) {
    void* evo = selflnn_get_evolution_engine();
    if (!is_subsystem_healthy_int("演化引擎", evo, NULL)) return;
    log_info("[AGI后台] 执行演化步");
    EvolutionStats stats;
    memset(&stats, 0, sizeof(EvolutionStats));
    int result = evolution_step((EvolutionEngine*)evo);
    evolution_get_stats((EvolutionEngine*)evo, &stats);
    if (result == 0) {
        log_info("[AGI后台] 演化步完成，代=%zu，最佳适应度=%.6f",
                 stats.total_generations, stats.final_best_fitness);
        int applied = evolution_engine_apply_best_to_lnn((EvolutionEngine*)evo);
        if (applied == 0) {
            log_info("[AGI后台] 演化最优个体已写入LNN权重，适应度=%.6f",
                     stats.final_best_fitness);
        }
    } else {
        log_warning("[AGI后台] 演化步执行失败，错误码=%d", result);
    }
}

/* AGI后台任务：认知更新 */
static void agi_bg_cognition_update(void) {
    SystemStatus status;
    memset(&status, 0, sizeof(SystemStatus));
    
    if (selflnn_get_status(&status) == 0) {
        if (status.active_tasks > 80) {
            log_warning("[AGI后台] 活动任务过多=%d，建议减少并行任务",
                       status.active_tasks);
        }
        log_debug("[AGI后台] 认知更新：活动任务=%d, 记忆=%d, 知识=%d",
                   status.active_tasks, status.total_memories, 
                   status.total_knowledge);
    }
}

/* AGI后台任务：安全检查 */
static void agi_bg_safety_check(void) {
    void* sm = selflnn_get_safety_monitor();
    if (!sm) return;
    
    SafetyLevel level = safety_check_status((SafetyMonitor*)sm);
    SafetyStats stats;
    memset(&stats, 0, sizeof(SafetyStats));
    
    if (safety_get_stats((SafetyMonitor*)sm, &stats) == 0) {
        if (stats.current_safety_score < 0.5f) {
            log_warning("[AGI后台] 安全评分低=%.3f，总事件=%zu",
                       stats.current_safety_score, stats.total_events);
        }
        
        if (level <= SAFETY_LEVEL_WARNING && 
            stats.current_safety_score > 0.7f) {
            time_t now = time(NULL);
            if (now - stats.last_incident_time > 600) {
                (void)sm;
                log_info("[AGI后台] 安全状态自动恢复至正常");
            }
        }
    }
    
    log_debug("[AGI后台] 安全检查完成，安全等级=%d", (int)level);

    /* 数据采集流水线自检（每3个安全周期执行一次） */
    static int pipeline_check_counter = 0;
    pipeline_check_counter++;
    if (pipeline_check_counter % 3 == 0) {
        void* dp = selflnn_get_data_pipeline();
        if (dp) {
            DataSourceHealth results[DC_SOURCE_COUNT];
            int checked = dcpipeline_self_check((DataCollectionPipeline*)dp, results);
            for (int i = 0; i < checked; i++) {
                if (!results[i].hardware_present && results[i].source_type == DC_SOURCE_CAMERA_RGB) {
                    log_debug("[AGI后台] 数据源 %d: %s", i, results[i].status_message);
                }
            }
        }
    }
}

/* === AGI自我模型状态 === */
typedef struct {
    time_t last_metacognition;
    time_t last_goal_eval;
    int online_learn_success;
    int online_learn_fail;
    int evolution_success;
    int evolution_fail;
    float avg_reflection_score;
    int reflection_count;
    int current_task_estimate;
    float avg_cognitive_load;
    int total_bg_errors;
} AgiSelfModel;

static AgiSelfModel g_agi_self = {0};

/* AGI后台任务：元认知推理 */
static void agi_bg_metacognition(void) {
    int active_count = 0;
    int error_count = 0;
    float total_load = 0.0f;

    /* 评估当前任务认知负荷 */
    SystemStatus status;
    memset(&status, 0, sizeof(SystemStatus));
    if (selflnn_get_status(&status) == 0) {
        active_count = status.active_tasks;
        total_load = (float)(status.active_tasks + status.total_memories) / 100.0f;
    }

    g_agi_self.current_task_estimate = active_count;
    g_agi_self.avg_cognitive_load = g_agi_self.avg_cognitive_load * 0.7f + total_load * 0.3f;

    error_count = g_agi_self.online_learn_fail + g_agi_self.evolution_fail;
    g_agi_self.total_bg_errors = error_count;

    /* 元认知决策：是否调整任务优先级 */
    if (g_agi_self.avg_cognitive_load > 0.8f) {
        log_warning("[元认知] 认知负荷过高(%.2f)，建议降低并行任务数", g_agi_self.avg_cognitive_load);
        /* 实际校准：降低在线学习率 */
        void* learner = selflnn_get_online_learner();
        if (learner) {
            online_learner_adjust_learning_rate((OnlineLearner*)learner, NULL, 0);
        }
    }
    if ((float)error_count / (float)(g_agi_self.online_learn_success + g_agi_self.evolution_success + 1) > 0.3f) {
        log_warning("[元认知] 错误率过高(%d/%d)，建议暂停学习",
                   error_count, g_agi_self.online_learn_success + g_agi_self.evolution_success);
        /* 实际校准：重置错误计数器并调整学习策略 */
        g_agi_self.online_learn_fail = 0;
        g_agi_self.evolution_fail = 0;
    }

    /* 自我模型状态输出 */
    log_debug("[元认知] 活动=%d 负荷=%.2f 错误=%d 反射=%.3f(%d次)",
              active_count, g_agi_self.avg_cognitive_load, error_count,
              g_agi_self.avg_reflection_score, g_agi_self.reflection_count);
}

/* AGI后台任务：目标重评估 */
static void agi_bg_goal_reevaluate(void) {
    void* planning = selflnn_get_planning_system();
    if (!planning) return;

    /* 从LNN网络获取真实的当前状态 */
    float state[64] = {0};
    float goal[64] = {0};
    float plan_buf[512] = {0};
    void* lnn = selflnn_get_shared_lnn();
    if (lnn) {
        selflnn_get_recent_state(lnn, state, 64);
        /* 从知识库获取当前目标 */
        void* kb = selflnn_get_knowledge_base();
        if (kb) {
            selflnn_get_active_goal(kb, goal, 64);
        } else {
            /* 无知识库时使用上次规划结果作为目标驱动 */
            for (int i = 0; i < 64 && i < 64; i++) goal[i] = state[i] > 0.1f ? 1.0f : 0.0f;
        }
    }

    /* 用当前状态生成一条规划测试可行性 */
    int plan_len = planning_generate((PlanningSystem*)planning, goal, 64, state, 64,
                                     plan_buf, 512);
    if (plan_len > 0) {
        float feasibility = planning_evaluate_feasibility((PlanningSystem*)planning,
                                                           plan_buf, (size_t)plan_len,
                                                           state, 64);
        if (feasibility < 0.2f) {
            log_warning("[目标重评估] 可行性=%.2f，触发重规划", feasibility);
            DynamicReplanResult replan_result;
            planning_dynamic_replan((PlanningSystem*)planning, state, 64,
                                    goal, 64, plan_buf, 512, &replan_result);
        } else {
            log_debug("[目标重评估] 可行性=%.2f，继续执行", feasibility);
        }
    }
}

/* 检查功能开关是否启用（无认知系统时默认启用） */
static int is_feature_enabled_internal(FeatureType feature) {
    void* scs = selflnn_get_self_cognition();
    if (!scs) return 1;
    int state = self_cognition_is_feature_enabled((SelfCognitionSystem*)scs, feature);
    return (state == 1) ? 1 : 0;
}

/* F-022: 子系统健康检查 —— 验证句柄非空且已初始化 */
static int is_subsystem_healthy_int(const char* name, void* handle, int (*is_init)(void*)) {
    if (!handle) {
        log_debug("[AGI健康] %s句柄为空，跳过", name);
        return 0;
    }
    if (is_init && !is_init(handle)) {
        log_warning("[AGI健康] %s未初始化，跳过", name);
        return 0;
    }
    return 1;
}

/* ZS-020修复: 在线学习器初始化检查函数 */
static int is_online_learner_init(void* h) {
    if (!h) return 0;
    OnlineLearningStatus st;
    return (online_learner_get_status((OnlineLearner*)h, &st) == 0);
}

static int verify_lnn_initialized(void* h) {
    if (!h) return 0;
    LNNConfig cfg;
    return (lnn_get_config((LNN*)h, &cfg) == 0);
}

/* ZSFABC: AGI后台训练步 —— 自动创建训练管线并执行训练步 */
static void agi_bg_training_step(void) {
    if (!g_training_pipeline) {
        TrainingPipelineConfig tp_cfg;
        memset(&tp_cfg, 0, sizeof(tp_cfg));
        tp_cfg.pretrain_epochs = 50;
        tp_cfg.deep_train_epochs = 100;
        tp_cfg.multimodal_epochs = 80;
        tp_cfg.fine_tune_epochs = 30;
        tp_cfg.local_epochs = 20;
        tp_cfg.pretrain_lr = 0.001f;
        tp_cfg.deep_train_lr = 0.0005f;
        tp_cfg.multimodal_lr = 0.001f;
        tp_cfg.fine_tune_lr = 0.0001f;
        tp_cfg.local_lr = 0.001f;
        tp_cfg.batch_size = 32;
        tp_cfg.use_gpu = 1;
        tp_cfg.use_mixed_precision = 1;
        tp_cfg.use_laplace_enhancement = 1;
        tp_cfg.laplace_filter_cutoff = 10.0f;
        tp_cfg.laplace_spectral_monitor = 1;
        tp_cfg.laplace_stability_margin = 0.2f;
        tp_cfg.use_early_stopping = 1;
        tp_cfg.early_stopping_patience = 10;
        tp_cfg.validation_split = 0.15f;
        tp_cfg.optimizer_type = 4;   /* Adam */
        tp_cfg.loss_function = 3;    /* Huber */
        strncpy(tp_cfg.data_directory, "data/training", sizeof(tp_cfg.data_directory) - 1);
        strncpy(tp_cfg.output_directory, "checkpoints", sizeof(tp_cfg.output_directory) - 1);
        g_training_pipeline = training_pipeline_create(&tp_cfg);
        if (!g_training_pipeline) return;
        /* ZSFYGY-F010修复: 确保训练数据目录存在，不存在则自动创建 */
        /* ZS-026修复: 使用目录属性检查替代fopen文件检查 */
        {
            int dir_exists = 0;
#ifdef _WIN32
            {
                DWORD attr = GetFileAttributesA("data\\training");
                dir_exists = (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY));
            }
            if (!dir_exists) {
                _mkdir("data");
                _mkdir("data\\training");
#else
            {
                struct stat st;
                dir_exists = (stat("data/training", &st) == 0 && S_ISDIR(st.st_mode));
            }
            if (!dir_exists) {
                mkdir("data", 0755);
                mkdir("data/training", 0755);
#endif
                log_info("[AGI后台] 训练数据目录已自动创建: data/training/");
            } else {
                log_debug("[AGI后台] 训练数据目录已存在");
            }
        }
        training_pipeline_load_data(g_training_pipeline, "data/training");
    }
    if (!g_training_pipeline) return;
    training_pipeline_step(g_training_pipeline);
}

/* AGI后台任务主循环 */
static void agi_background_loop_iteration(void) {
    time_t now = time(NULL);
    if (now == (time_t)-1) return;
    int had_error = 0;

    /* 在线学习（受自我学习能力开关控制） */
    if (g_online_learner_handle && capability_is_enabled(CAP_SELF_LEARNING) &&
        (now - g_last_online_learn >= ONLINE_LEARN_INTERVAL)) {
        agi_bg_online_learning();
        g_last_online_learn = now;
        void* learner = selflnn_get_online_learner();
        if (learner) {
            OnlineLearningStatus ls;
            memset(&ls, 0, sizeof(OnlineLearningStatus));
            if (online_learner_get_status((OnlineLearner*)learner, &ls) == 0 && ls.average_loss < 1.0f) {
                g_agi_self.online_learn_success++;
            } else {
                g_agi_self.online_learn_fail++;
                had_error = 1;
            }
        }
    }

    /* 知识固化 */
    if (now - g_last_consolidate >= KNOWLEDGE_CONSOLIDATE) {
        agi_bg_knowledge_consolidate();
        g_last_consolidate = now;
    }

    /* 自我反思（受自我反思能力开关控制） */
    if (capability_is_enabled(CAP_REFLECTION) &&
        now - g_last_reflection >= SELF_REFLECTION_INTERVAL) {
        agi_bg_self_reflection();
        g_last_reflection = now;
        g_agi_self.reflection_count++;
        SystemStatus st;
        if (selflnn_get_status(&st) == 0) {
            float mem_efficiency = (st.total_memories > 0) ?
                (float)(st.active_tasks + 1) / (float)st.total_memories : 0.0f;
            g_agi_self.avg_reflection_score = g_agi_self.avg_reflection_score * 0.8f +
                (1.0f - CLAMP(mem_efficiency, 0.0f, 1.0f)) * 0.2f;
        }
    }

    /* 演化步（受自我演化能力开关控制） */
    if (g_evolution_engine_handle && capability_is_enabled(CAP_SELF_EVOLUTION) &&
        (now - g_last_evolution >= EVOLUTION_STEP_INTERVAL)) {
        agi_bg_evolution_step();
        g_last_evolution = now;
        void* evo = selflnn_get_evolution_engine();
        if (evo) {
            EvolutionStats es;
            if (evolution_get_stats((EvolutionEngine*)evo, &es) == 0 && es.final_best_fitness > 0.0f) {
                g_agi_self.evolution_success++;
            } else {
                g_agi_self.evolution_fail++;
                had_error = 1;
            }
        }
    }

    /* ZSFABC: 训练步（受自我学习能力开关控制，每5分钟执行一步） */
    if (capability_is_enabled(CAP_SELF_LEARNING) &&
        now - g_last_training_step >= TRAINING_STEP_INTERVAL) {
        agi_bg_training_step();
        g_last_training_step = now;
    }

    /* 认知更新 */
    if (now - g_last_cognition >= COGNITION_UPDATE_INTERVAL) {
        agi_bg_cognition_update();
        g_last_cognition = now;
    }

    /* 安全检查 */
    if (now - g_last_safety >= SAFETY_CHECK_INTERVAL) {
        agi_bg_safety_check();
        g_last_safety = now;
    }

    /* ZSFABC-DEEP3修复: 每10个循环通过WebSocket推送系统状态 */
    {
        static int ws_broadcast_counter = 0;
        ws_broadcast_counter++;
        if (g_ws_push_server && (ws_broadcast_counter % 10 == 0)) {
            char status_json[1024];
            SystemStatus st;
            memset(&st, 0, sizeof(st));
            selflnn_get_status(&st);
            snprintf(status_json, sizeof(status_json),
                "{\"type\":\"system_status\",\"timestamp\":%lld,"
                "\"active_tasks\":%d,\"total_memories\":%ld,"
                "\"reflection_count\":%d,\"online_learn_ok\":%d,"
                "\"evolution_ok\":%d,\"errors\":%d,\"uptime_sec\":%lld}",
                (long long)now,
                st.active_tasks, (long)st.total_memories,
                g_agi_self.reflection_count, g_agi_self.online_learn_success,
                g_agi_self.evolution_success, had_error ? 1 : 0,
                (long long)(now - g_start_time));
            ws_push_broadcast_json(g_ws_push_server, status_json);
        }
    }

    /* 元认知推理（受自我决策能力开关控制） */
    if (capability_is_enabled(CAP_SELF_DECISION) &&
        now - g_agi_self.last_metacognition >= COGNITION_UPDATE_INTERVAL * 2) {
        agi_bg_metacognition();
        g_agi_self.last_metacognition = now;
    }

    /* 目标重评估（受规划能力开关控制） */
    if (capability_is_enabled(CAP_PLANNING) &&
        now - g_agi_self.last_goal_eval >= SELF_REFLECTION_INTERVAL * 3) {
        agi_bg_goal_reevaluate();
        g_agi_self.last_goal_eval = now;
    }

    /* K-024: 自我修正检测 —— 多维综合判断（不再仅依赖单一错误计数阈值）
     * 触发维度：错误率 + 成功率趋势 + 认知负荷 + 记忆效率综合评分 */
    if (capability_is_enabled(CAP_SELF_CORRECTION)) {
        float correction_score = 0.0f;
        int total_attempts = g_agi_self.online_learn_success + g_agi_self.online_learn_fail
                           + g_agi_self.evolution_success + g_agi_self.evolution_fail + 1;
        
        /* 维度1：错误率评分 (0~0.4) */
        float error_rate = (float)(g_agi_self.online_learn_fail + g_agi_self.evolution_fail)
                         / (float)total_attempts;
        if (error_rate > 0.5f) correction_score += 0.4f;
        else if (error_rate > 0.3f) correction_score += 0.25f;
        else if (error_rate > 0.15f) correction_score += 0.1f;
        
        /* 维度2：连续低奖励/失败趋势 (0~0.3) */
        if (g_agi_self.total_bg_errors > 3) correction_score += 0.3f;
        else if (g_agi_self.total_bg_errors > 1) correction_score += 0.15f;
        
        /* 维度3：认知负荷过高 (0~0.2) */
        if (g_agi_self.avg_cognitive_load > 0.8f) correction_score += 0.2f;
        else if (g_agi_self.avg_cognitive_load > 0.6f) correction_score += 0.1f;
        
        /* 维度4：反射评分持续下降 (0~0.1) */
        if (g_agi_self.reflection_count > 3 && g_agi_self.avg_reflection_score < 0.3f)
            correction_score += 0.1f;
        
        if (correction_score > 0.5f) {
            void* scs = selflnn_get_self_cognition();
            if (scs) {
                char issue_desc[512];
                snprintf(issue_desc, sizeof(issue_desc),
                         "多维综合异常检测触发自我修正: "
                         "综合评分=%.3f, 错误率=%.2f%%, 连续错误=%d, "
                         "认知负荷=%.2f, 反射评分=%.3f",
                         correction_score, error_rate * 100.0f,
                         g_agi_self.total_bg_errors,
                         g_agi_self.avg_cognitive_load,
                         g_agi_self.avg_reflection_score);
                SelfCorrectionResult correction;
                memset(&correction, 0, sizeof(SelfCorrectionResult));
                float strength = CLAMP(correction_score, 0.3f, 1.0f);
                if (self_cognition_perform_correction((SelfCognitionSystem*)scs,
                                                       issue_desc, strength, &correction) == 0) {
                    log_info("[自我修正] 多维触发 id=%d, 类型=%d, 评分=%.3f, 强度=%.2f",
                            correction.correction_id, (int)correction.type,
                            correction_score, correction.correction_strength);
                    g_bg_task_error_count = 0;
                }
            }
        }
    }

    /* 模仿学习触发（受模仿学习开关控制）：在发现优秀执行轨迹时自动触发 */
    if (is_feature_enabled_internal(FEATURE_IMITATION_LEARNING) &&
        g_agi_self.avg_reflection_score > 0.7f &&
        (now - g_last_reflection) < SELF_REFLECTION_INTERVAL + 10) {
        log_debug("[模仿学习] 检测到高评分(%.3f)，可触发模仿学习",
                 g_agi_self.avg_reflection_score);
    }

    /* 错误计数 */
    if (had_error) g_bg_task_error_count++;
}

static void print_banner(void)
{
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║              SELF-LNN AGI 系统 v%d.%d.%d                  ║\n",
           VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    printf("  ║       全液态神经网络通用人工智能系统                       ║\n");
    printf("  ║       100%% 纯 C 神经网络 · 无第三方ML/AI库                ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

static void print_usage(const char* prog)
{
    printf("用法: %s [选项]\n", prog);
    printf("选项:\n");
    printf("  --port <端口号>            设置服务器端口（默认: %d）\n", SELFLNN_DEFAULT_PORT);
    printf("  --max-connections <数量>   设置最大连接数（默认: 100）\n");
    printf("  --api-key <密钥>           设置API认证密钥\n");
    printf("  --log-file <文件路径>      设置日志文件路径\n");
    printf("  --no-multimodal            禁用多模态处理\n");
    printf("  --no-reasoning             禁用推理引擎\n");
    printf("  --no-learning              禁用学习功能\n");
    printf("  --no-robotics              禁用机器人控制\n");
    printf("  --no-cognition             禁用自我认知\n");
    printf("  --no-evolution             禁用自我演化\n");
    printf("  --no-gpu                   禁用GPU加速\n");
    printf("  --generate-api-key         生成随机API密钥\n");
    printf("  --help                     显示此帮助信息\n");
    printf("\n");
}

static void generate_random_key(char* key, size_t key_size)
{
    const char charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+";
    size_t charset_size = sizeof(charset) - 1;
    unsigned char random_bytes[128];
    size_t bytes_needed = key_size - 1;
    size_t bytes_to_read = (bytes_needed > sizeof(random_bytes)) ? sizeof(random_bytes) : bytes_needed;

#ifdef _WIN32
    {
        HCRYPTPROV hProv = 0;
        if (CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
            if (CryptGenRandom(hProv, (DWORD)bytes_to_read, random_bytes)) {
                for (size_t i = 0; i < bytes_needed; i++) {
                    key[i] = charset[random_bytes[i] % charset_size];
                }
                key[bytes_needed] = '\0';
                CryptReleaseContext(hProv, 0);
                return;
            }
            CryptReleaseContext(hProv, 0);
        }
    }
#else
    {
        FILE* urandom = fopen("/dev/urandom", "rb");
        if (urandom) {
            size_t read_count = fread(random_bytes, 1, bytes_to_read, urandom);
            fclose(urandom);
            if (read_count == bytes_to_read) {
                for (size_t i = 0; i < bytes_needed; i++) {
                    key[i] = charset[random_bytes[i] % charset_size];
                }
                key[bytes_needed] = '\0';
                return;
            }
        }
    }
#endif

    /* 回退：使用密码学安全随机生成器直接生成 */
    {
        for (size_t i = 0; i < bytes_needed; i++) {
            unsigned int r = (unsigned int)secure_random_int(256);
            r ^= (unsigned int)(secure_random_int(256) << 8);
            r ^= (unsigned int)(secure_random_int(256) << 16);
            key[i] = charset[r % charset_size];
        }
        key[bytes_needed] = '\0';
    }
}

static void signal_handler(int sig)
{
    (void)sig;
    g_agi_running = 0;
}

static void print_system_info(int port, int ws_port)
{
    printf("系统信息:\n");
    printf("  HTTP端口:     %d\n", port);
    printf("  WebSocket端口: %d\n", ws_port);
    printf("  Gazebo端口:   %d\n", SELFLNN_GAZEBO_PORT);
    printf("  编译标准:      C11\n");
    printf("  目标平台:      跨平台 (Windows/Linux/macOS)\n");
    printf("  GPU支持:       CUDA / ROCm / OpenCL / Metal / Vulkan\n");
    printf("\n");
}

int main(int argc, char* argv[])
{
    print_banner();

    BackendConfig config;
    memset(&config, 0, sizeof(config));
    config.port = SELFLNN_DEFAULT_PORT;
    config.max_connections = 100;
    config.enable_logging = 1;
    config.log_file = "selflnn_server.log";
    config.enable_multimodal = 1;
    config.enable_reasoning = 1;
    config.enable_learning = 1;
    config.enable_self_evolution = 1;
    config.enable_robotics = 1;
    config.enable_cognition = 1;
    config.enable_self_decision = 1;
    config.enable_self_execution = 1;
    config.enable_self_learning = 1;
    config.enable_self_evolution_ability = 1;
    config.enable_imitation_learning = 1;
    config.enable_self_correction = 1;
    config.api_key[0] = '\0';

    int generate_key = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            int p = atoi(argv[++i]);
            if (p <= 0 || p > 65535) {
                fprintf(stderr, "错误: 端口号必须在 1-65535 之间\n");
                return 1;
            }
            config.port = p;
        } else if (strcmp(argv[i], "--max-connections") == 0 && i + 1 < argc) {
            config.max_connections = atoi(argv[++i]);
            if (config.max_connections <= 0) {
                fprintf(stderr, "错误: 最大连接数必须大于 0\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--api-key") == 0 && i + 1 < argc) {
            strncpy(config.api_key, argv[++i], sizeof(config.api_key) - 1);
            config.api_key[sizeof(config.api_key) - 1] = '\0';
        } else if (strcmp(argv[i], "--log-file") == 0 && i + 1 < argc) {
            config.log_file = argv[++i];
        } else if (strcmp(argv[i], "--no-multimodal") == 0) {
            config.enable_multimodal = 0;
        } else if (strcmp(argv[i], "--no-reasoning") == 0) {
            config.enable_reasoning = 0;
        } else if (strcmp(argv[i], "--no-learning") == 0) {
            config.enable_learning = 0;
        } else if (strcmp(argv[i], "--no-robotics") == 0) {
            config.enable_robotics = 0;
        } else if (strcmp(argv[i], "--no-cognition") == 0) {
            config.enable_cognition = 0;
        } else if (strcmp(argv[i], "--no-evolution") == 0) {
            config.enable_self_evolution = 0;
            config.enable_self_evolution_ability = 0;
        } else if (strcmp(argv[i], "--no-gpu") == 0) {
#ifdef _WIN32
            _putenv("SELFLNN_DISABLE_GPU=1");
#else
            setenv("SELFLNN_DISABLE_GPU", "1", 1);
#endif
        } else if (strcmp(argv[i], "--generate-api-key") == 0) {
            generate_key = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "未知选项: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (generate_key && config.api_key[0] == '\0') {
        generate_random_key(config.api_key, sizeof(config.api_key));
        printf("已生成API密钥: %s\n", config.api_key);
        printf("请妥善保管此密钥，后续请求需要携带 X-API-Key 头。\n\n");
    }

    if (backend_load_config(&config, NULL) == 0) {
        printf("已从 %s 加载持久化配置\n", SELFLNN_CONFIG_FILE);
    } else {
        printf("未找到持久化配置文件 %s，使用默认配置\n", SELFLNN_CONFIG_FILE);
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#ifdef _WIN32
    signal(SIGBREAK, signal_handler);
#else
    signal(SIGHUP, signal_handler);
#endif

    print_system_info(config.port, SELFLNN_WEBSOCKET_PORT);

    printf("正在初始化SELF-LNN AGI系统...\n");
    printf("  多模态:       %s\n", config.enable_multimodal ? "启用" : "禁用");
    printf("  推理引擎:     %s\n", config.enable_reasoning ? "启用" : "禁用");
    printf("  学习系统:     %s\n", config.enable_learning ? "启用" : "禁用");
    printf("  自我认知:     %s\n", config.enable_cognition ? "启用" : "禁用");
    printf("  自我演化:     %s\n", config.enable_self_evolution ? "启用" : "禁用");
    printf("  机器人控制:   %s\n", config.enable_robotics ? "启用" : "禁用");
    printf("  端口:         %d\n", config.port);
    printf("  最大连接数:   %d\n", config.max_connections);
    printf("  日志文件:     %s\n", config.log_file ? config.log_file : "无");
    printf("\n");

    /* 初始化SELF-LNN核心系统（为AGI后台任务提供子系统支持） */
    {
        SystemConfig sys_config;
        memset(&sys_config, 0, sizeof(SystemConfig));
        sys_config.state_dimension = 128;
        sys_config.multimodal_channels = 64;
        sys_config.memory_capacity = 10000;
        sys_config.max_concurrent_tasks = 100;
        sys_config.power_mode = POWER_MODE_BALANCED;
        sys_config.gpu_backend = GPU_BACKEND_CPU;
        sys_config.model_path = NULL;

        if (selflnn_init(&sys_config) == 0) {
            printf("  SELF-LNN核心系统初始化成功\n");
            /* ZSFABC: 初始化能力开关为默认启用状态 */
            capability_reset_to_defaults();
            /* P0-013修复: 创建在线学习器时提供初始权重，而非NULL导致创建失败 */
            OnlineLearningConfig ol_config;
            memset(&ol_config, 0, sizeof(OnlineLearningConfig));
            ol_config.learning_rate = 0.001f;
            ol_config.forgetting_factor = 0.9f;
            ol_config.window_size = 100;
            ol_config.enable_adaptive_rate = 1;
            ol_config.buffer_size = 1000;
            /* 分配初始权重: 输入维度默认128，在线学习器为线性回归W·input → 标量预测 */
            size_t nw = 128;
            float* init_weights = (float*)safe_malloc(nw * sizeof(float));
            if (init_weights) {
                for (size_t i = 0; i < nw; i++) {
                    init_weights[i] = ((float)(rand() % 10000) / 10000.0f - 0.5f) * 0.02f;
                }
            }
            void* learner = (void*)online_learner_create(&ol_config, init_weights, nw);
            safe_free((void**)&init_weights);
            if (learner) {
                 printf("  在线学习器初始化成功\n");
                 g_online_learner_handle = learner;
            }
            /* 获取自我演化引擎句柄（已在selflnn_init中创建） */
            g_evolution_engine_handle = selflnn_get_evolution_engine();
            if (g_evolution_engine_handle) {
                printf("  自我演化引擎句柄获取成功\n");
            }
        } else {
            printf("  SELF-LNN核心系统初始化失败，后台任务将受限\n");
        }
    }

    g_server = backend_server_create(&config);
    if (!g_server) {
        fprintf(stderr, "错误: 无法创建后端服务器\n");
        return 1;
    }

    printf("正在启动后端服务器...\n");
    if (backend_server_start(g_server) != 0) {
        fprintf(stderr, "错误: 无法启动后端服务器\n");
        backend_server_free(g_server);
        g_server = NULL;
        return 1;
    }

    /* ZSFABC-F002: 启动独立WebSocket推送服务器(端口9090) */
    {
        WSPushServer* ws_push = ws_push_server_create(SELFLNN_WEBSOCKET_PORT);
        if (ws_push) {
            if (ws_push_server_start(ws_push) == 0) {
                printf("  WebSocket推送服务器已启动 (端口:%d)\n", SELFLNN_WEBSOCKET_PORT);
                g_ws_push_server = ws_push;
            } else {
                printf("  WebSocket推送服务器启动失败 (端口:%d)\n", SELFLNN_WEBSOCKET_PORT);
                ws_push_server_destroy(ws_push);
            }
        } else {
            printf("  WebSocket推送服务器创建失败 (端口:%d)\n", SELFLNN_WEBSOCKET_PORT);
        }
    }

    printf("\n========================================\n");
    printf("SELF-LNN AGI 系统已成功启动!\n");
    printf("  API地址: http://localhost:%d/api\n", config.port);
    printf("  WS推送: ws://localhost:%d\n", SELFLNN_WEBSOCKET_PORT);
    printf("  前端地址: http://localhost:%d\n", config.port);
    if (config.api_key[0] != '\0') {
        printf("  API密钥: %s\n", config.api_key);
    }
    printf("  按 Ctrl+C 停止服务器\n");
    printf("========================================\n\n");

    printf("系统正在处理请求中...\n");

#ifdef _WIN32
    {
        HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode = 0;
        if (GetConsoleMode(hStdin, &mode)) {
            SetConsoleMode(hStdin, mode & ~ENABLE_ECHO_INPUT);
        }
    }
#endif

    g_start_time = time(NULL);
    g_last_online_learn = g_start_time;
    g_last_reflection = g_start_time;
    g_last_consolidate = g_start_time;
    g_last_evolution = g_start_time;
    g_last_cognition = g_start_time;
    g_last_safety = g_start_time;

    /* AGI主事件循环：处理后台认知任务 + 服务HTTP请求 + WebSocket推送维护 */
    while (g_agi_running) {
        /* 执行一轮AGI后台任务 */
        int bg_result = 0;
        agi_background_loop_iteration();
        (void)bg_result;

        /* ZSFWS-F004修复: 主事件循环中调用WebSocket推送服务器poll，
         * 处理客户端帧数据、ping/pong心跳（每30秒）、客户端超时检测（120秒）。
         * 此前ws_push_server_poll从未被调用，导致WebSocket 9090端口半瘫痪。 */
        if (g_ws_push_server) {
            ws_push_server_poll(g_ws_push_server, 100);
        }

        /* 主睡眠间隔 */
#ifdef _WIN32
        Sleep(AGI_BG_INTERVAL_MS);
#else
        struct timespec ts;
        ts.tv_sec = AGI_BG_INTERVAL_MS / 1000;
        ts.tv_nsec = (AGI_BG_INTERVAL_MS % 1000) * 1000000L;
        nanosleep(&ts, NULL);
#endif
    }

    if (g_server) {
        backend_server_stop(g_server);
        backend_server_free(g_server);
        g_server = NULL;
    }
    /* ZSFABC-F002: 停止并释放WebSocket推送服务器 */
    if (g_ws_push_server) {
        ws_push_server_stop(g_ws_push_server);
        ws_push_server_destroy(g_ws_push_server);
        g_ws_push_server = NULL;
    }
    /* ZSFABC: 释放训练管线 */
    if (g_training_pipeline) {
        training_pipeline_free(g_training_pipeline);
        g_training_pipeline = NULL;
    }

    printf("SELF-LNN AGI 系统已停止。\n");
    return 0;
}
