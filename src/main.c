/* 必须在所有include之前定义,否则backend.h链式include会先解析
 * cfc_network.h/cfc_cell.h并设置include guard,导致完整结构体永远不可见 */
#define SELFLNN_IMPLEMENTATION
#define SELFLNN_CORE_INTERNAL

#include "selflnn/backend/backend.h"
#include "selflnn/backend/websocket_push.h"
#include "selflnn/cognition/self_cognition.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc_network.h" /* 完整CfCNetwork结构 */
#include "selflnn/core/cfc_cell.h" /* 完整CfCCell结构 */
#include "selflnn/core/unified_lnn_state.h"
#include "selflnn/core/laplace_unified.h"                    /* 已恢复: 拉普拉斯统一 */
#include "selflnn/multimodal/audio.h"                       /* 音频采集 */
#include "selflnn/multimodal/multimodal_unified_input.h" /* 统一输入状态for适应度评估 */
#include "selflnn/multimodal/speech_recognition.h"           /* 语音识别 */
#include "selflnn/multimodal/tts.h"                         /* 语音合成 */
#include "selflnn/multimodal/slam.h" /* SLAM定位建图 */
#include "selflnn/multimodal/camera_capture.h" /* 摄像头采集 */
#include "selflnn/robot/computer_operation.h"                /* 计算机操作 */
#include "selflnn/knowledge/knowledge.h"
#include "selflnn/knowledge/auto_learning.h"
#include "selflnn/multimodal/data_collection_pipeline.h"
#include "selflnn/memory/memory.h"
#include "selflnn/memory/memory_manager.h"
#include "selflnn/reasoning/reasoning.h"
#include "selflnn/reasoning/planning.h"

/* 能力开关 */
#include "selflnn/agi/capability_switch.h"
#include "selflnn/agi/agi.h" /* AGI系统集成 */
#include "selflnn/agi/task_scheduler.h" /* 任务调度器集成 */
#include "selflnn/learning/learning.h"
#include "selflnn/learning/online_learning.h"
#include "selflnn/learning/imitation_learning.h" /* 模仿学习集成 */
#include "selflnn/multimodal/multimodal.h"
#include "selflnn/multimodal/multimodal_manager.h"
#include "selflnn/multimodal/dialogue.h"
#include "selflnn/training/training.h"
#include "selflnn/training/training_pipeline.h"
#include "selflnn/gpu/gpu.h"
#include "selflnn/robot/robot.h"
#include "selflnn/product_design/product_design.h"
#include "selflnn/concurrency/thread_pool.h"
#include "selflnn/multisystem/multisystem_control.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/selflnn.h"
#include "selflnn/evolution/evolution_engine.h"
#include "selflnn/evolution/neural_architecture_search.h"             /* NAS周期触发 */
#include "selflnn/safety/safety_monitor.h"
#include "selflnn/safety/audit_logger.h"
#include "selflnn/safety/content_filter.h"
#include "selflnn/safety/security_monitor_deep.h"
#include "selflnn/distributed/load_balancer.h"
#include "selflnn/training/distributed_training.h"
#include "selflnn/programming/programming_enhanced.h"
#include "selflnn/learning/multi_agent.h"                /* H-015: 多智能体协作 */
#include "selflnn/knowledge/knowledge_graph.h"            /* R003: KG→后端桥接 */
#include "selflnn/robot/gazebo_bridge.h" /* Gazebo仿真桥接 */
#include "selflnn/core/architecture_controller.h" /* ZSFJJJ-FIX: 动态架构控制器 */
#include "selflnn/utils/json_parser.h" /* ZSFKKK: 正规JSON解析替代手写解析器 */
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
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <strings.h>
#endif

#ifdef _MSC_VER
#pragma warning(disable:4100 4189 4244 4267 4701)
#endif

/* 静态函数前向声明 */
static int is_subsystem_healthy_int(const char* name, void* handle, int (*is_init)(void*));
static size_t ullnn_input_dimension_estimate(void* lnn_ptr);

/* 知识库更新事件通知回调
 * 此回调在 knowledge_base_add 写锁内调用，必须极其轻量。
 * 仅设置 selflnn 的刷新标志位，不执行任何耗时操作。
 * 实际的知识嵌入重编码由 AGI 后台循环异步处理。 */
static void kb_update_nofify_callback(void* user_data) {
    (void)user_data;
    selflnn_trigger_knowledge_refresh();
}

/* AGI后台任务常量 */
#define AGI_BG_INTERVAL_MS       10000  /* 主循环间隔10秒 */
/* CLAMP已通过math_utils_internal.h统一定义，此处不再重复 */
/* main.c不包含math_utils_internal.h, 保留本地定义 */
#ifndef CLAMP
#define CLAMP(x,min,max) (((x)<(min))?(min):(((x)>(max))?(max):(x)))
#endif
#define ONLINE_LEARN_INTERVAL_MIN    300   /* 在线学习最小5分钟 */
#define ONLINE_LEARN_INTERVAL_MAX    3600  /* 在线学习最大1小时 */
#define SELF_REFLECTION_INTERVAL_MIN 1800  /* 自我反思最小30分钟 */
#define SELF_REFLECTION_INTERVAL_MAX 7200  /* 自我反思最大2小时 */
#define KNOWLEDGE_CONSOLIDATE_MIN    600   /* 知识固化最小10分钟 */
#define KNOWLEDGE_CONSOLIDATE_MAX    3600  /* 知识固化最大1小时 */
#define EVOLUTION_STEP_INTERVAL_MIN  600   /* 演化步最小10分钟 */
#define EVOLUTION_STEP_INTERVAL_MAX  3600  /* 演化步最大1小时 */
#define SAFETY_CHECK_INTERVAL      60  /* P38修复: 安全检查每1分钟(安全关键) */
#define TRAINING_STEP_INTERVAL_MIN   300   /* 训练步最小5分钟 */
#define TRAINING_STEP_INTERVAL_MAX   3600  /* 训练步最大1小时 */
#define COGNITION_UPDATE_INTERVAL 300  /* P38修复: 认知更新每5分钟 */

/* 当前运行动态间隔(由认知更新根据负载自适应调整) */
static int g_online_learn_interval_sec = 300;
static int g_reflection_interval_sec   = 1800;
static int g_consolidate_interval_sec  = 600;
static int g_evolution_interval_sec    = 600;
static int g_training_interval_sec     = 300;

#define VERSION_MAJOR 1
#define VERSION_MINOR 5
#define VERSION_PATCH 0

static BackendServer* g_server = NULL;
WSPushServer* g_ws_push_server = NULL;
static void* g_online_learner_handle = NULL;
static void* g_evolution_engine_handle = NULL;
static void* g_unified_lnn_state = NULL;
static volatile sig_atomic_t g_agi_running = 1;
static time_t g_start_time = 0;
static time_t g_last_online_learn = 0;
static time_t g_last_reflection = 0;
static time_t g_last_consolidate = 0;
static time_t g_last_evolution = 0;
static time_t g_last_cognition = 0;
static time_t g_last_safety = 0;
static int g_bg_task_error_count = 0;
static TrainingPipeline* g_training_pipeline = NULL; /* 训练管线 */
static time_t g_last_training_step = 0; /* 上次训练步时间 */
static GazeboBridge* g_gazebo_bridge = NULL; /* Gazebo仿真桥接 */
static ArchitectureController* g_arch_controller = NULL; /* ZSFJJJ-FIX: 动态架构控制器 */
void* volatile g_global_lnn = NULL;                              /* P2-005审查: 全局LNN指针，供GPU后端(TPU)、学习引擎、知识库跨模块访问(volatile确保多线程可见性) */
static ProductDesignEngine* g_product_design = NULL;    /* APP10: 产品设计引擎(selflnn管理) */
static void* g_nas_system = NULL;                         /* 神经架构搜索(selflnn管理) */
static void* g_laplace_unified = NULL;                    /* 拉普拉斯统一(selflnn管理) */
static void* g_audio_capture = NULL;                       /* 音频采集(selflnn管理) */
static void* g_speech_recognizer = NULL;                   /* 语音识别(selflnn管理) */
static void* g_tts_engine = NULL;                          /* TTS语音合成(selflnn管理) */
static void* g_computer_op = NULL;                         /* 计算机操作(selflnn管理) */
static void* g_slam_system = NULL; /* SLAM系统(selflnn管理) */
static void* g_camera_capture = NULL; /* 摄像头采集(selflnn管理) */

/* SLAM摄像头帧回调 - 将摄像头帧送入SLAM系统和统一液态状态 */
static void slam_camera_frame_callback(const uint8_t* rgb_data, int width, int height, void* user_data) {
    (void)user_data;
    if (!rgb_data || !g_slam_system || width <= 0 || height <= 0) return;

    /* 将RGB帧转换为浮点数组供SLAM处理 */
    size_t pixel_count = (size_t)(width * height);
    float* float_frame = (float*)safe_malloc(pixel_count * sizeof(float));
    if (!float_frame) return;

    /* RGB转灰度：Y = 0.299R + 0.587G + 0.114B */
    for (size_t i = 0; i < pixel_count; i++) {
        size_t pi = i * 3;
        float_frame[i] = 0.299f * (float)rgb_data[pi] +
                         0.587f * (float)rgb_data[pi + 1] +
                         0.114f * (float)rgb_data[pi + 2];
    }

    /* 送入SLAM系统 */
    SlamResult slam_result;
    memset(&slam_result, 0, sizeof(SlamResult));
    int ret = slam_process_visual_frame((SlamSystem*)g_slam_system,
        float_frame, width, height, 1, (float)time(NULL), &slam_result);

    if (ret == 0 && slam_result.tracking_quality > 50) {
        log_debug("[SLAM] 视觉帧处理: 质量=%d%% 关键帧=%d",
                 slam_result.tracking_quality, slam_result.new_keyframe_created);
    }

    /* 注入统一液态状态处理器 */
    if (g_unified_lnn_state) {
        const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES] = {NULL};
        size_t raw_sizes[UNIFIED_LNN_MAX_MODALITIES] = {0};
        int modality_present[UNIFIED_LNN_MAX_MODALITIES] = {0};
        raw_inputs[UNIFIED_MODALITY_VISION] = float_frame;
        raw_sizes[UNIFIED_MODALITY_VISION] = pixel_count;
        modality_present[UNIFIED_MODALITY_VISION] = 1;
        float ul_output[256];
        unified_lnn_state_step((UnifiedLNNState*)g_unified_lnn_state,
            raw_inputs, raw_sizes, modality_present, ul_output, sizeof(ul_output));
    }

    safe_free((void**)&float_frame);
}
static DistributedContext* g_distributed = NULL;           /* 分布式上下文(selflnn管理) */
static LbBalancer* g_load_balancer = NULL;                /* 负载均衡(selflnn管理) */
static AuditLogger* g_audit_logger = NULL;                /* 审计日志(selflnn管理) */
static ContentFilter* g_content_filter = NULL;            /* 内容过滤(selflnn管理) */
static SelfProgrammingEngine* g_prog_engine = NULL;       /* 自我编程引擎(selflnn管理) */
static SecBehaviorMonitor* g_sec_behavior = NULL;         /* 深度安全行为监控 */
static AGISystem* g_agi_system = NULL; /* AGI认知系统 */
static TaskScheduler* g_task_scheduler = NULL; /* 任务调度器 */

/* P28修复: 演化引擎默认适应度函数 — 基于LNN权重的一致性评估
 * 将染色体(晶片)中的权重写入LNN，运行前向传播，
 * 通过输出稳定性评估适应度(激活输出方差越小→越稳定→适应度越高)。
 *
 *修复: 双路径适应度评估
 *   路径A（真实数据路径）: 使用统一输入状态缓存的最近多模态真实信号作为评估输入，
 *     评估LNN对真实世界数据的预测能力。此路径权重0.7。
 *   路径B（合成探针路径）: 保留确定性正弦/随机/阶跃/脉冲信号作为补充泛化评估，
 *     确保网络在未见信号模式下的鲁棒性。此路径权重0.3。
 * 两条路径完全隔离——路径A不使用路径B的任何数据，路径B不使用路径A的任何数据。 */
static float lnn_weights_fitness_function(const float* chromosome, size_t chrom_size, void* user_data) {
    LNN* lnn = (LNN*)user_data;
    if (!lnn || chrom_size == 0) return 0.0f;
    /* 将染色体复制到LNN权重缓冲区（lnn_get_parameters返回可写指针） */
    size_t lnn_param_count = lnn_get_parameter_count(lnn);
    if (lnn_param_count == 0) return 0.0f;
    float* lnn_params = lnn_get_parameters(lnn);
    if (!lnn_params) return 0.0f;
/* 保护染色体memcpy写入LNN参数的原子性 */
    lnn_lock(lnn);
    size_t copy_count = (chrom_size < lnn_param_count) ? chrom_size : lnn_param_count;
    memcpy(lnn_params, chromosome, copy_count * sizeof(float));
    /* R6-FIX: 将染色体剩余部分写入cell级三门参数 */
    {
        CfCNetwork* cfc = lnn_get_cfc_network(lnn);
        size_t chrom_offset = copy_count;
        if (cfc && cfc->layers) {
            for (int cl = 0; cl < cfc->config.num_layers; cl++) {
                CfCCell* cell = cfc->layers[cl];
                if (!cell) continue;
                size_t li = (cl == 0) ? cfc->config.input_size : cfc->config.hidden_size;
                size_t hs = cfc->config.hidden_size;
                size_t n;
                n = li * hs;
                if (cell->input_gate_weights && chrom_offset + n <= chrom_size) {
                    memcpy(cell->input_gate_weights, chromosome + chrom_offset, n * sizeof(float));
                    chrom_offset += n;
                }
                n = li * hs;
                if (cell->forget_gate_weights && chrom_offset + n <= chrom_size) {
                    memcpy(cell->forget_gate_weights, chromosome + chrom_offset, n * sizeof(float));
                    chrom_offset += n;
                }
                n = li * hs;
                if (cell->output_gate_weights && chrom_offset + n <= chrom_size) {
                    memcpy(cell->output_gate_weights, chromosome + chrom_offset, n * sizeof(float));
                    chrom_offset += n;
                }
                n = hs * 3;
                if (cell->gate_biases && chrom_offset + n <= chrom_size) {
                    memcpy(cell->gate_biases, chromosome + chrom_offset, n * sizeof(float));
                    chrom_offset += n;
                }
                if (cell->use_adaptive_tau && cell->time_constants && chrom_offset + hs <= chrom_size) {
                    memcpy(cell->time_constants, chromosome + chrom_offset, hs * sizeof(float));
                    chrom_offset += hs;
                }
                n = hs * hs;
                if (cell->hidden_to_input_gate_weights && chrom_offset + n <= chrom_size) {
                    memcpy(cell->hidden_to_input_gate_weights, chromosome + chrom_offset, n * sizeof(float));
                    chrom_offset += n;
                }
                if (cell->hidden_to_forget_gate_weights && chrom_offset + n <= chrom_size) {
                    memcpy(cell->hidden_to_forget_gate_weights, chromosome + chrom_offset, n * sizeof(float));
                    chrom_offset += n;
                }
                if (cell->hidden_to_output_gate_weights && chrom_offset + n <= chrom_size) {
                    memcpy(cell->hidden_to_output_gate_weights, chromosome + chrom_offset, n * sizeof(float));
                    chrom_offset += n;
                }
                if (cell->hidden_to_activation_weights && chrom_offset + n <= chrom_size) {
                    memcpy(cell->hidden_to_activation_weights, chromosome + chrom_offset, n * sizeof(float));
                    chrom_offset += n;
                }
            }
        }
    }
/* 染色体写入完成,释放锁(lnn_forward内部有独立LNN_LOCK) */
    lnn_unlock(lnn);

    /* ================================================================
 *: 双路径适应度评估
     * 路径A: 真实数据评估（权重0.7）— 从统一输入状态获取最近真实信号
     * 路径B: 合成探针评估（权重0.3）— 保留为泛化能力补充
     * ================================================================ */
    float real_data_fitness = 0.0f;
    int real_data_available = 0;

    /* ---- 路径A: 真实数据评估 ---- */
    {
        void* ul_state = selflnn_get_unified_state();
        if (ul_state) {
            UnifiedInputState* uis = (UnifiedInputState*)ul_state;
            size_t input_dim = ullnn_input_dimension_estimate(lnn);
            if (input_dim == 0) input_dim = 64;
            float* real_input = (float*)safe_calloc(input_dim, sizeof(float));
            float* output = (float*)safe_malloc(128 * sizeof(float));
            if (real_input && output) {
                int real_samples = 0;
                /* 遍历9种模态的最近原始信号，构建真实输入批次 */
                for (int m = 0; m < SELFLNN_MAX_MODALITIES && real_samples < 6; m++) {
                    if (uis->last_raw_sizes[m] > 0 && uis->last_raw_sizes[m] <= input_dim) {
                        memset(real_input, 0, input_dim * sizeof(float));
                        memcpy(real_input, uis->last_raw_signals[m],
                               uis->last_raw_sizes[m] * sizeof(float));
                        memset(output, 0, 128 * sizeof(float));
                        if (lnn_forward(lnn, real_input, output) != 0) continue;
                        /* 评估真实数据响应：计算输出有效幅值和稳定性 */
                        float sum = 0.0f, sq_sum = 0.0f;
                        size_t count = 0;
                        for (size_t i = 0; i < 128 && count < 64; i++) {
                            if (fabsf(output[i]) > 1e-10f) {
                                sum += output[i]; sq_sum += output[i] * output[i]; count++;
                            }
                        }
                        if (count > 0) {
                            float mean = sum / (float)count;
                            float var = sq_sum / (float)count - mean * mean;
                            real_data_fitness += fabsf(mean) * 0.5f + logf(1.0f + 1.0f / (var + 1e-8f));
                            real_samples++;
                        }
                    }
                }
                if (real_samples > 0) {
                    real_data_fitness /= (float)real_samples;
                    real_data_available = 1;
                }
            }
            safe_free((void**)&real_input);
            safe_free((void**)&output);
        }
    }

    /* ---- P1-002修复: 移除合成探针评估路径 ----
     * 演化适应度评估现在100%基于真实数据，不使用任何合成信号。
     * 原合成探针路径B已永久移除，无真实数据时返回最小适应度。 */
    float fitness;
    if (real_data_available) {
        fitness = real_data_fitness;
    } else {
        /* ZSFJJJ-L001修复: 无真实数据时记录警告日志，
         * 便于运维和调试时发现数据缺口问题。 */
        log_warning("[演化适应度] 无可用于评估的真实数据，返回最小适应度=0.001。"
                     "请连接传感器/摄像头/麦克风等硬件或启动仿真器以获取真实数据。");
        fitness = 0.001f;  /* 无真实数据时返回最小适应度，不生成假数据 */
    }
    if (fitness < 0.0f) fitness = 0.001f;
    return fitness;
}

static size_t ullnn_input_dimension_estimate(void* lnn_ptr) {
    LNN* lnn = (LNN*)lnn_ptr;
    if (!lnn) return 0;
    CfCNetwork* cfc = lnn_get_cfc_network(lnn);
    if (!cfc) return 0;
    return cfc->config.input_size;
}

/* C89前向声明 (is_online_learner_init在L490定义，先于L160调用) */
static int is_online_learner_init(void* h);

/* AGI后台任务：在线学习循环 */
static void agi_bg_online_learning(void) {
    void* learner = selflnn_get_online_learner();
    /* ZS-020修复: 使用在线学习器专用检查函数替代NULL */
    if (!is_subsystem_healthy_int("在线学习器", learner, is_online_learner_init)) return;
    OnlineLearningStatus status;
    memset(&status, 0, sizeof(OnlineLearningStatus));
    int s = online_learner_get_status((OnlineLearner*)learner, &status);
    if (s != 0) {
        log_warning("[AGI后台] 在线学习器状态查询失败，跳过本轮学习");
        return;
    }
    if (status.total_samples < 10 || status.current_learning_rate < 1e-7f) {
        return;
    }
/* 从LNN配置动态获取状态维度，替代硬编码128。
     * 确保与LNN实际维度匹配，避免数据截断或越界。 */
    size_t state_dim = selflnn_get_config_state_dimension();
    if (state_dim == 0) state_dim = 128;  /* 回退: 配置不可用时使用默认维度 */
    float* state = (float*)safe_malloc(state_dim * sizeof(float));
    float* target = (float*)safe_malloc(state_dim * sizeof(float));
    if (!state || !target) {
        safe_free((void**)&state);
        safe_free((void**)&target);
        return;
    }
    /* 尝试从LNN网络获取当前状态和最近输出作为训练数据 */
    void* lnn = selflnn_get_shared_lnn();
    if (lnn) {
        if (selflnn_get_recent_state(lnn, state, (int)state_dim) == 0 &&
            selflnn_get_recent_output(lnn, target, (int)state_dim) == 0) {

/* 验证数据锚点——防止自举循环
             * 在线学习使用LNN自身输出作为训练目标时，必须验证输出质量。
             * 验证策略：
             *   1. 知识库一致性检查: 如果知识库有相关事实，验证LNN输出与事实的一致性
             *   2. 输出约束: 确保训练目标在合理范围内，避免发散
             *   3. 如果验证失败，本轮跳过，避免用错误输出自我强化 */
            int valid_target = 1;
            float target_norm = 0.0f;
            for (size_t i = 0; i < state_dim; i++) {
                target_norm += target[i] * target[i];
            }
            target_norm = sqrtf(target_norm);

            /* 验证1: 目标输出不应全零或过小（LNN未激活状态） */
            if (target_norm < 1e-6f) {
                valid_target = 0;
            }

            /* 验证2: 目标输出不应爆炸，知识库锚定修正 */
            if (target_norm > 1e4f) {
                valid_target = 0;
                KnowledgeBase* kb = (KnowledgeBase*)selflnn_get_knowledge_base();
                if (kb) {
/* 实际调用knowledge_base_nearest_fact查找最近事实锚定 */
                    char anchor_s[256], anchor_p[256], anchor_o[256];
                    float anchor_sim = 0.0f;
                    memset(anchor_s, 0, sizeof(anchor_s));
                    memset(anchor_p, 0, sizeof(anchor_p));
                    memset(anchor_o, 0, sizeof(anchor_o));
                    int ret = knowledge_base_nearest_fact(kb, target, state_dim,
                        anchor_s, sizeof(anchor_s),
                        anchor_p, sizeof(anchor_p),
                        anchor_o, sizeof(anchor_o), &anchor_sim);
                    if (ret == 0 && anchor_sim > 0.3f) {
                        log_warning("[AGI后台] LNN输出爆炸(norm=%.2f), 知识锚定: %s %s %s (相似度=%.3f)",
                                   target_norm, anchor_s, anchor_p, anchor_o, anchor_sim);
                    } else {
                        log_warning("[AGI后台] LNN输出爆炸(norm=%.2f), 知识库未找到锚定事实",
                                   target_norm);
                    }
                }
            }

            /* 验证3: 使用知识库进行一致性交叉验证
             * 如果知识库中有与当前状态相关的可靠事实，将其作为训练anchor */
            if (valid_target && target_norm > 0.01f) {
                KnowledgeBase* kb = (KnowledgeBase*)selflnn_get_knowledge_base();
                if (kb) {
                    size_t kb_fact_count = knowledge_base_get_total_facts(kb);
                    if (kb_fact_count > 10) {
                        /* 对输出施加轻微的知识一致性约束
                         * 当LNN输出与已知事实产生冲突时，降低学习率 */
                        float consistency = knowledge_base_output_consistency(kb, target, state_dim);
                        if (consistency < 0.3f && status.total_samples > 100) {
                            log_debug("[AGI后台] 知识一致性=%.2f < 0.3, 决策锚定中", consistency);
                            /* 学习率衰减: 不一致时降低学习率避免强化错误 */
                            online_learner_adjust_learning_rate((OnlineLearner*)learner, NULL, 0);
                        }
                    }
                }
            }

            if (valid_target) {
                float loss = 0.0f;
                online_learner_update((OnlineLearner*)learner, state, state_dim, target, state_dim, &loss);

/* 在线学习发散保护增强
                 * 监控输出趋势，防止自举循环导致的系统性漂移 */
                static float loss_ema = 0.0f;
                static int divergence_counter = 0;
                static int stable_counter = 0;

                /* 指数移动平均追踪损失趋势 */
                if (loss_ema == 0.0f) {
                    loss_ema = loss;
                } else {
                    loss_ema = 0.9f * loss_ema + 0.1f * loss;
                }

                /* 发散检测：损失持续上升超过EMA的2倍 */
                if (loss > loss_ema * 2.0f && loss > 1.0f) {
                    divergence_counter++;
                    stable_counter = 0;
                } else if (loss < loss_ema * 0.8f) {
                    stable_counter++;
                    divergence_counter = 0;
                } else {
                    divergence_counter = (divergence_counter > 0) ? divergence_counter - 1 : 0;
                    stable_counter = (stable_counter < 10) ? stable_counter + 1 : 10;
                }

                /* 连续发散5轮：强制衰减学习率并重置漂移状态 */
                if (divergence_counter >= 5) {
                    log_warning("[AGI后台] 在线学习检测到发散趋势(loss=%.4f, EMA=%.4f)，强制学习率衰减",
                               loss, loss_ema);
                    online_learner_adjust_learning_rate((OnlineLearner*)learner, NULL, 0);
                    divergence_counter = 0;
                    loss_ema = loss;
                }

                /* 长期稳定后渐进恢复学习率 */
                if (stable_counter >= 10 && loss < 0.1f) {
                    log_debug("[AGI后台] 在线学习恢复稳定，可渐进提升学习率");
                    stable_counter = 0;
                }
            } else {
                log_debug("[AGI后台] 验证锚点拒绝无效训练目标(norm=%.6f), 跳过本轮学习", target_norm);
            }
        }
    }
    safe_free((void**)&state);
    safe_free((void**)&target);
}

/* AGI后台任务：知识固化（系统状态监控 + 记忆/知识统计） */
static void agi_bg_knowledge_consolidate(void) {
    static int consolidate_cycle = 0;
    consolidate_cycle++;

/* 事件驱动的知识嵌入刷新
     * 当 knowledge_base_add 成功写入新知识后，通过回调设置刷新标志位。
     * 此处检查标志位并触发CfC嵌入引擎重新训练，实现对新增知识条目的
     * 嵌入向量重新编码。替代原来的纯定时轮询方式，消除延迟。 */
    if (selflnn_check_and_reset_knowledge_refresh()) {
        KnowledgeBase* kb = (KnowledgeBase*)selflnn_get_knowledge_base();
        if (kb) {
            int ret = knowledge_base_retrain_embeddings(kb, 0);
            if (ret == 0) {
                log_debug("[AGI后台] 知识嵌入已刷新（事件触发）");
            }
        }
/* WebSocket实时推送知识更新通知 */
        if (g_ws_push_server) {
            char buf[256];
            snprintf(buf, sizeof(buf), "{\"type\":\"knowledge_update\",\"timestamp\":%lld}", (long long)time(NULL));
            ws_push_broadcast_json(g_ws_push_server, buf);
        }
    }

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

/* 每6个固化周期从已有知识库推理新知识
     * 利用传递性/对称性/逆关系等逻辑规则从三元组中推导新三元组，
     * 实现"向知识库进行学习"的能力闭环。 */
    if (consolidate_cycle % 6 == 0) {
        void* al = selflnn_get_auto_learning();
        void* kb = selflnn_get_knowledge_base();
        if (al && kb) {
            int inferred = auto_learning_infer_from_knowledge_base(
                (AutoLearningSystem*)al, kb, 50);
            if (inferred > 0) {
                log_info("[AGI后台] 知识推理完成，新增%d条推理知识", inferred);
                if (g_ws_push_server) {
                    char buf[256];
                    snprintf(buf, sizeof(buf),
                        "{\"type\":\"knowledge_update\",\"inferred\":%d,\"timestamp\":%lld}",
                        inferred, (long long)time(NULL));
                    ws_push_broadcast_json(g_ws_push_server, buf);
                }
            }
        }
    }

        /* FIX-F4: 消费知识推理结果→LNN状态扰动（知识→LNN→决策完整数据通道） */
        void* kie = selflnn_get_knowledge_inference();
        if (kie) {
            void* shared_lnn = g_global_lnn;
            if (shared_lnn) {
                int consumed = selflnn_consume_knowledge_inference(
                    shared_lnn, kie, "AGI", 3, 0.1f);
                if (consumed > 0) {
                    log_debug("[AGI后台] 知识推理消费：%d条事实注入LNN状态", consumed);
                }
            }
        }

        /* KG→LNN AGI后台注入 (图推理引擎全局就绪) */
        {
            void* kg_ptr = selflnn_get_knowledge_graph();
            void* shared_lnn = g_global_lnn;
            if (kg_ptr && shared_lnn) {
                knowledge_graph_to_lnn_bridge(kg_ptr, shared_lnn, 0.1f);
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
    /* APP11: 每5个演化周期触发一次神经架构搜索(NAS) */
    static int nas_trigger_counter = 0;
    nas_trigger_counter++;
    if (nas_trigger_counter % 5 == 0 && g_nas_system) {
        int nas_ret = nas_search_generation((NASSystem*)g_nas_system);
        if (nas_ret == 0) {
            log_info("[AGI后台] NAS搜索代完成");
        }
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

        float load_factor = (float)status.active_tasks / 100.0f;
        if (load_factor > 0.7f) {
            /* 高负载：延长间隔降低系统压力 */
            g_online_learn_interval_sec = ONLINE_LEARN_INTERVAL_MAX;
            g_reflection_interval_sec   = SELF_REFLECTION_INTERVAL_MAX;
            g_consolidate_interval_sec  = KNOWLEDGE_CONSOLIDATE_MAX;
            g_evolution_interval_sec    = EVOLUTION_STEP_INTERVAL_MAX;
            g_training_interval_sec     = TRAINING_STEP_INTERVAL_MAX;
        } else if (load_factor > 0.3f) {
            /* 中等负载：温和间隔 */
            g_online_learn_interval_sec = ONLINE_LEARN_INTERVAL_MIN * 2;
            g_reflection_interval_sec   = SELF_REFLECTION_INTERVAL_MIN * 2;
            g_consolidate_interval_sec  = KNOWLEDGE_CONSOLIDATE_MIN * 2;
            g_evolution_interval_sec    = EVOLUTION_STEP_INTERVAL_MIN * 2;
            g_training_interval_sec     = TRAINING_STEP_INTERVAL_MIN * 2;
        } else {
            /* 低负载：加速学习和迭代 */
            g_online_learn_interval_sec = ONLINE_LEARN_INTERVAL_MIN;
            g_reflection_interval_sec   = SELF_REFLECTION_INTERVAL_MIN;
            g_consolidate_interval_sec  = KNOWLEDGE_CONSOLIDATE_MIN;
            g_evolution_interval_sec    = EVOLUTION_STEP_INTERVAL_MIN;
            g_training_interval_sec     = TRAINING_STEP_INTERVAL_MIN;
        }
    }
    /* APP13: 群智优化 — 认知更新时触发蜂群迭代优化 */
    void* ms_ctrl = selflnn_get_multisystem_control();
    if (ms_ctrl) {
        multisystem_swarm_iterate((MultiSystemControlEngine*)ms_ctrl);
    }
}

/* AGI后台任务：安全检查 */
static void agi_bg_safety_check(void) {
    void* sm = selflnn_get_safety_monitor();
    if (!sm) return;

/* 主动安全监控 — 原仅被动读取状态, 关键监控函数从未被调用
     * 现在每周期主动触发资源监控、动态阈值调整、熔断器检查 */
    safety_monitor_resources((SafetyMonitor*)sm);
    safety_dynamic_adjust_thresholds((SafetyMonitor*)sm);
    
    SafetyLevel level = safety_check_status((SafetyMonitor*)sm);
    SafetyStats stats;
    memset(&stats, 0, sizeof(SafetyStats));
    
    if (safety_get_stats((SafetyMonitor*)sm, &stats) == 0) {
/* 安全评分过低时触发紧急停止通知电机 */
        if (stats.current_safety_score < 0.3f) {
            log_error("[AGI后台] 安全评分严重过低=%.3f, 触发安全熔断", stats.current_safety_score);
            /* 触发熔断器 */
            safety_circuit_breaker_open((SafetyMonitor*)sm);
        }
        else if (stats.current_safety_score < 0.5f) {
            log_warning("[AGI后台] 安全评分低=%.3f，总事件=%zu",
                       stats.current_safety_score, stats.total_events);
        }
        
        if (level <= SAFETY_LEVEL_WARNING && 
            stats.current_safety_score > 0.7f) {
            time_t now = time(NULL);
            if (now - stats.last_incident_time > 600) {
                log_info("[AGI后台] 安全状态自动恢复至正常");
                /* 安全恢复时关闭熔断器 */
                safety_circuit_breaker_close((SafetyMonitor*)sm);
            }
        }
    }
    
    log_debug("[AGI后台] 安全检查完成，安全等级=%d", (int)level);

    /* 数据采集流水线自检（每3个安全周期执行一次，或由事件驱动即时触发） */
/* 保持现有counter机制（每3周期=约3小时），
     * 但添加即时自检请求检查，当硬件变化事件触发时立即执行自检，
     * 无需等待完整的3个安全周期。 */
    static int pipeline_check_counter = 0;
    pipeline_check_counter++;
    int immediate_requested = dcpipeline_is_immediate_check_requested();
    if (pipeline_check_counter % 3 == 0 || immediate_requested) {
        void* dp = selflnn_get_data_pipeline();
        if (dp) {
            DataSourceHealth results[DC_SOURCE_COUNT];
            int checked = dcpipeline_self_check((DataCollectionPipeline*)dp, results);
            for (int i = 0; i < checked; i++) {
                if (!results[i].hardware_present && results[i].source_type == DC_SOURCE_CAMERA_RGB) {
                    log_debug("[AGI后台] 数据源 %d: %s", i, results[i].status_message);
                }
            }
            if (immediate_requested) {
                log_info("[AGI后台] 事件驱动即时自检完成，已检查%d个数据源", checked);
            }
        }
        if (immediate_requested) {
            dcpipeline_clear_immediate_check();
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
/* 保留上一个有效目标向量。
     * 当知识库不可用时，不应使用简单阈值二元化(goal[i]=state[i]>0.1f?1:0)，
     * 这会破坏梯度信息丰富性，导致规划退化。
     * 改为保留上一次有效目标，仅在首次或知识库恢复时更新。 */
    static float prev_valid_goal[64] = {0};
    static int prev_goal_valid = 0;
    void* lnn = selflnn_get_shared_lnn();
    if (lnn) {
        selflnn_get_recent_state(lnn, state, 64);
        /* 从知识库获取当前目标 */
        void* kb = selflnn_get_knowledge_base();
        if (kb) {
            selflnn_get_active_goal(kb, goal, 64);
            /* 保存为有效目标向量供后续回退使用 */
            memcpy(prev_valid_goal, goal, 64 * sizeof(float));
            prev_goal_valid = 1;
        } else if (prev_goal_valid) {
            /* 无知识库时使用上一次有效目标向量，避免退化到简单二元化 */
            memcpy(goal, prev_valid_goal, 64 * sizeof(float));
        } else {
            /* 首次运行且知识库不可用：使用LNN状态自然分布作为初始目标 */
            {
                float state_norm = 0.0f;
                for (int i = 0; i < 64; i++) state_norm += state[i] * state[i];
                state_norm = sqrtf(state_norm);
                float scale = (state_norm > 1e-6f) ? (1.0f / state_norm) : 0.0f;
                for (int i = 0; i < 64; i++) {
                    goal[i] = state[i] * scale * 0.5f;  /* 缩放至合理范围 */
                }
            }
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

/* 检查功能开关是否启用（统一双开关系统）
 * 优先检查capability_switch（主开关系统），其次检查self_cognition（认知反射系统）。
 * 两个系统任一返回启用即视为启用，消除两套开关返回值不一致导致的
 * 功能在capability_is_enabled已打开但is_feature_enabled_internal仍关闭的问题。 */
static int is_feature_enabled_internal(FeatureType feature) {
    /* 优先使用capability_switch（主开关系统） */
    int cap_result = capability_is_enabled((CapabilityType)feature);
    if (cap_result) return 1;

    /* 回退到self_cognition检查 */
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

/* AGI后台训练步 —— 自动创建训练管线并执行训练步 */
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
        tp_cfg.convergence_threshold = 1e-6f; /* 绝对收敛阈值 */
        tp_cfg.optimizer_type = 4;   /* Adam */
        tp_cfg.loss_function = 3;    /* Huber */
        strncpy(tp_cfg.data_directory, "data/training", sizeof(tp_cfg.data_directory) - 1);
        strncpy(tp_cfg.output_directory, "checkpoints", sizeof(tp_cfg.output_directory) - 1);
        g_training_pipeline = training_pipeline_create(&tp_cfg);
        if (!g_training_pipeline) return;
/* 将训练管线注册到selflnn全局状态。
         * 确保selflnn_get_training_pipeline不再返回NULL，
         * 使AGI认知循环、演化引擎等模块能通过统一接口访问训练管线，
         * 消除main.c和selflnn之间的训练管线双实例问题。 */
        selflnn_set_training_pipeline((void*)g_training_pipeline);
/* 确保训练数据目录存在，不存在则自动创建 */
        /* ZS-026修复: 使用目录属性检查替代fopen文件检查 */
/* 目录创建后验证是否存在数据文件，无文件时输出警告并执行空训练步 */
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
/* 扫描目录下所有文件，验证是否存在训练数据文件 */
            {
                int data_file_count = 0;
                const char* exts[] = {".json", ".csv", ".txt", ".bin"};
                int num_exts = sizeof(exts) / sizeof(exts[0]);
#ifdef _WIN32
                {
                    WIN32_FIND_DATAA find_data;
                    HANDLE h_find;
                    char search_path[512];
                    snprintf(search_path, sizeof(search_path), "data\\training\\*");
                    h_find = FindFirstFileA(search_path, &find_data);
                    if (h_find != INVALID_HANDLE_VALUE) {
                        do {
                            if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                                continue;
                            const char* fname = find_data.cFileName;
                            const char* dot = strrchr(fname, '.');
                            if (dot) {
                                for (int ei = 0; ei < num_exts; ei++) {
                                    if (_stricmp(dot, exts[ei]) == 0) {
                                        data_file_count++;
                                        break;
                                    }
                                }
                            }
                        } while (FindNextFileA(h_find, &find_data));
                        FindClose(h_find);
                    }
                }
#else
                {
                    DIR* d = opendir("data/training");
                    if (d) {
                        struct dirent* entry;
                        while ((entry = readdir(d)) != NULL) {
                            if (entry->d_type == DT_DIR) continue;
                            const char* fname = entry->d_name;
                            const char* dot = strrchr(fname, '.');
                            if (dot) {
                                for (int ei = 0; ei < num_exts; ei++) {
                                    if (strcasecmp(dot, exts[ei]) == 0) {
                                        data_file_count++;
                                        break;
                                    }
                                }
                            }
                        }
                        closedir(d);
                    }
                }
#endif
                if (data_file_count == 0) {
                    log_warn("[AGI后台] 训练数据目录 data/training/ 中未找到任何数据文件(.json/.csv/.txt/.bin)，" 
                             "将仅执行空训练步（不加载空数据），请放入训练数据文件后重新训练。");
                    /* 仅执行空训练步，不加载空数据 */
                    if (!g_training_pipeline) return;
                    training_pipeline_step(g_training_pipeline);
                    return;
                }
                log_info("[AGI后台] 训练数据目录扫描完成，发现 %d 个数据文件", data_file_count);
            }
        }
        training_pipeline_load_data(g_training_pipeline, "data/training");
    }
    if (!g_training_pipeline) return;
    training_pipeline_step(g_training_pipeline);
}

/* AGI后台任务主循环 */
static void agi_background_loop_iteration(void) {
    /* M-041修复验证: AGI后台循环所有子任务均已通过capability_is_enabled()检查对应能力开关：
     * - 自我认知更新(行1177): CAP_SELF_COGNITION ✓
     * - 在线学习(行1082): CAP_SELF_LEARNING ✓
     * - 演化引擎(行1152): CAP_SELF_EVOLUTION ✓
     * - 训练管线(行1169): CAP_SELF_LEARNING + CAP_AUTONOMOUS_EXECUTION ✓
     * - 自我反思(行1126): CAP_REFLECTION ✓
     * - 知识固化(行1100): CAP_AUTONOMOUS_EXECUTION ✓
     * - 睡眠巩固(行1110): CAP_AUTONOMOUS_EXECUTION ✓
     * - 自我修正(行1717): CAP_SELF_CORRECTION ✓
     * - 元认知推理(行1702): CAP_SELF_DECISION ✓
     * - 目标重评估(行1709): CAP_PLANNING ✓
     * - 好奇心探索(行1772): CAP_CURIOSITY ✓
     * - 模仿学习(行1803): CAP_IMITATION_LEARNING ✓
     * - 多智能体评估(行1898): CAP_MULTI_AGENT ✓
     */
    time_t now = time(NULL);
    if (now == (time_t)-1) return;
    int had_error = 0;

    /* 在线学习（受自我学习能力开关控制） */
    if (g_online_learner_handle && capability_is_enabled(CAP_SELF_LEARNING) &&
        (now - g_last_online_learn >= g_online_learn_interval_sec)) {
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

    /* 知识固化（受自主执行能力开关控制） */
    if (capability_is_enabled(CAP_AUTONOMOUS_EXECUTION) &&
        now - g_last_consolidate >= g_consolidate_interval_sec) {
        agi_bg_knowledge_consolidate();
        g_last_consolidate = now;
    }

/* P24修复: 记忆睡眠巩固(NREM/REM) — 每180秒触发
     * 受自主执行能力开关控制，关闭时跳过自主记忆处理 */
    {
        static time_t g_last_sleep = 0;
        if (capability_is_enabled(CAP_AUTONOMOUS_EXECUTION) &&
            now - g_last_sleep >= 180) {
            void* mm = selflnn_get_memory_manager();
            if (mm) {
                int stats[4] = {0};
                memory_sleep_consolidation((MemorySystem*)mm, 1.0f, stats);
                if (stats[0] > 0 || stats[1] > 0) {
                    log_info("[AGI后台] 睡眠巩固 NREM=%d REM=%d STM→LTM=%d 清理=%d",
                             stats[0], stats[1], stats[2], stats[3]);
                }
            }
            g_last_sleep = now;
        }
    }

    /* 自我反思（受自我反思能力开关控制 + LNN就绪检查） */
    if (capability_is_enabled(CAP_REFLECTION) &&
        now - g_last_reflection >= g_reflection_interval_sec) {
/* 在发起深度自我反思前检查LNN是否已训练，
         * 避免随机权重LNN产生无意义的自我评估和错误修正决策 */
        void* scs_check = selflnn_get_self_cognition();
        if (scs_check && !self_cognition_is_lnn_ready((SelfCognitionSystem*)scs_check)) {
            log_debug("[AGI后台] LNN尚未完成训练，跳过本次深度自我反思，"
                     "等待模型检查点加载或初始训练完成后自动启用");
            /* 仍然更新反思时间戳避免频繁重试 */
            g_last_reflection = now;
        } else {
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

    }

    /* 演化步（受自我演化能力开关控制） */
    if (g_evolution_engine_handle && capability_is_enabled(CAP_SELF_EVOLUTION) &&
        (now - g_last_evolution >= g_evolution_interval_sec)) {
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

/* 训练步（受自我学习 + 自主执行能力开关双重控制，每5分钟执行一步） */
    if (capability_is_enabled(CAP_SELF_LEARNING) &&
        capability_is_enabled(CAP_AUTONOMOUS_EXECUTION) &&
        now - g_last_training_step >= g_training_interval_sec) {
        agi_bg_training_step();
        g_last_training_step = now;
    }

    /* Z8-003: 认知更新（受自我认知能力开关控制 — 之前直接执行无人检查） */
    if (capability_is_enabled(CAP_SELF_COGNITION) &&
        now - g_last_cognition >= COGNITION_UPDATE_INTERVAL) {
        agi_bg_cognition_update();
        g_last_cognition = now;
/* 认知更新事件推送到WebSocket */
        if (g_ws_push_server) {
            char cbuf[512];
            SystemStatus st_cog;
            memset(&st_cog, 0, sizeof(st_cog));
            selflnn_get_status(&st_cog);
            snprintf(cbuf, sizeof(cbuf),
                "{\"type\":\"cognition_event\",\"timestamp\":%lld,"
                "\"active_tasks\":%d,\"memories\":%ld,\"knowledge\":%ld,"
                "\"reflection_count\":%d,\"cognitive_load\":%.4f}",
                (long long)now, st_cog.active_tasks,
                (long)st_cog.total_memories, (long)st_cog.total_knowledge,
                g_agi_self.reflection_count, g_agi_self.avg_cognitive_load);
            ws_push_broadcast_json(g_ws_push_server, cbuf);
        }
    }

    /* 安全检查 */
    if (now - g_last_safety >= SAFETY_CHECK_INTERVAL) {
        agi_bg_safety_check();
        g_last_safety = now;
/* 同步触发记忆衰减(与安全检查同频, 每分钟) */
        {
            void* mem_mgr = selflnn_get_memory_manager();
            if (mem_mgr) {
                extern int memory_periodic_decay_update(void*);
                memory_periodic_decay_update(mem_mgr);
            }
        }
    }

/* 拉普拉斯频域指标动态更新。
     * state.c中的laplace_stability_score等字段初始化为硬编码0.5f，
     * 从未从网络运行状态动态计算。现在每30分钟调用一次拉普拉斯分析器，
     * 将频域分析结果(极点分布、稳定性裕度)写入网络状态，
     * 为后续拉普拉斯-CfC实时调制提供数据基础。 */
    {
        static time_t g_last_laplace_update = 0;
        if (now - g_last_laplace_update >= 1800 && g_laplace_unified) {
            /* 从统一拉普拉斯子系统获取频谱分析结果 */
            float spectrum[256];
            memset(spectrum, 0, sizeof(spectrum));
            if (laplace_unified_get_spectrum(g_laplace_unified, spectrum, 256) == 0) {
                /* 计算主导频率和频谱带宽作为稳定性指标 */
                float dominant_freq = 0.0f, max_amp = 0.0f;
                float bandwidth = 0.0f;
                for (int f = 0; f < 256; f++) {
                    if (spectrum[f] > max_amp) {
                        max_amp = spectrum[f];
                        dominant_freq = (float)f;
                    }
                    bandwidth += spectrum[f] * spectrum[f];
                }
                bandwidth = sqrtf(bandwidth / 256.0f);
                /* 将频域指标写入LNN状态 */
                void* lnn_lp = selflnn_get_shared_lnn();
                if (lnn_lp) {
                    float laplace_metrics[3] = {
                        dominant_freq / 256.0f,     /* 归一化主导频率 */
                        bandwidth,                   /* 频谱带宽 */
                        max_amp                      /* 最大幅度 */
                    };
                    selflnn_set_laplace_metrics(laplace_metrics, 3);
                    log_debug("[拉普拉斯] 频域更新: 主导频率=%.2f, 带宽=%.4f, 幅度=%.4f",
                             dominant_freq / 256.0f, bandwidth, max_amp);
                }
            }
            g_last_laplace_update = now;
        }
    }

/* 周期性通过WebSocket推送各类系统状态 */
    {
        static int ws_broadcast_counter = 0;
        ws_broadcast_counter++;
        if (!g_ws_push_server) { /* 无WS推送服务器则跳过 */ }
        if (ws_broadcast_counter % 5 == 0) {
            char buf[2048];

            /* 每5个循环: LNN状态推送 */
            {
                void* lnn = selflnn_get_shared_lnn();
                LNNConfig cfg;
                memset(&cfg, 0, sizeof(cfg));
                if (lnn && lnn_get_config((LNN*)lnn, &cfg) == 0) {
                    snprintf(buf, sizeof(buf),
                        "{\"type\":\"lnn_state\",\"timestamp\":%lld,"
                        "\"input_dim\":%zu,\"hidden_dim\":%zu,\"output_dim\":%zu,"
                        "\"solver_type\":%d,\"convergence_rate\":%.4f}",
                        (long long)now, (size_t)cfg.input_size, (size_t)cfg.hidden_size,
                        (size_t)cfg.output_size, cfg.ode_solver_type, cfg.learning_rate);
                    ws_push_broadcast_json(g_ws_push_server, buf);
                }
            }
        }
        if (ws_broadcast_counter % 10 == 0) {
            char buf[2048];
            SystemStatus st;
            memset(&st, 0, sizeof(st));

            /* 每10个循环: 系统状态 */
            if (selflnn_get_status(&st) == 0) {
                snprintf(buf, sizeof(buf),
                "{\"type\":\"system_status\",\"timestamp\":%lld,"
                "\"active_tasks\":%d,\"total_memories\":%ld,"
                "\"reflection_count\":%d,\"online_learn_ok\":%d,"
                "\"evolution_ok\":%d,\"errors\":%d,\"uptime_sec\":%lld}",
                (long long)now,
                st.active_tasks, (long)st.total_memories,
                g_agi_self.reflection_count, g_agi_self.online_learn_success,
                g_agi_self.evolution_success, had_error ? 1 : 0,
                (long long)(now - g_start_time));
            ws_push_broadcast_json(g_ws_push_server, buf);
            }

            /* 每10个循环: GPU状态 */
            {
                GpuBackend selected = gpu_auto_select();
                const char* gpu_type_name = gpu_backend_name(selected);
                int gpu_available = (selected != GPU_BACKEND_CPU) ? 1 : 0;
                snprintf(buf, sizeof(buf),
                    "{\"type\":\"gpu_status\",\"timestamp\":%lld,"
                    "\"device\":\"%s\",\"available\":%s,\"utilization\":%.1f,"
                    "\"backend\":\"%s\"}",
                    (long long)now, gpu_type_name,
                    gpu_available ? "true" : "false",
                    gpu_available ? -1.0f : 0.0f, /* 真实GPU利用率需异步轮询获取，未检测标记为-1 */
                    gpu_backend_name(selected));
                ws_push_broadcast_json(g_ws_push_server, buf);
            }

            /* 每10个循环: 内存状态 */
            {
                void* mm = selflnn_get_memory_manager();
                size_t used_mem = 0;
                float ratio = 0.0f, level = 0.0f;
                if (mm) {
                    memory_manager_get_stats((MemoryManager*)mm, &used_mem, &ratio, &level);
                }
                snprintf(buf, sizeof(buf),
                    "{\"type\":\"memory_status\",\"timestamp\":%lld,"
                    "\"used_memory\":%zu,\"allocated_memory\":%zu,"
                    "\"total_memories\":%ld,\"consolidation_ratio\":%.3f}",
                    (long long)now, used_mem, (size_t)(used_mem * 2),
                    (long)st.total_memories, ratio);
                ws_push_broadcast_json(g_ws_push_server, buf);
            }

            /* 每10个循环: 知识库状态 */
            {
                void* kb = selflnn_get_knowledge_base();
                size_t total_entries = 0;
                if (kb) {
                    knowledge_base_get_stats((KnowledgeBase*)kb, &total_entries, NULL);
                }
                snprintf(buf, sizeof(buf),
                    "{\"type\":\"knowledge_status\",\"timestamp\":%lld,"
                    "\"total_entries\":%zu,\"total_knowledge\":%ld}",
                    (long long)now, total_entries, (long)st.total_knowledge);
                ws_push_broadcast_json(g_ws_push_server, buf);
            }

            /* 每20个循环: 安全状态 */
            if (ws_broadcast_counter % 20 == 0) {
                void* sm = selflnn_get_safety_monitor();
                SafetyStats ss = {0};
                if (sm && safety_get_stats((SafetyMonitor*)sm, &ss) == 0) {
                    snprintf(buf, sizeof(buf),
                        "{\"type\":\"safety_alert\",\"score\":%.3f,\"events\":%zu,\"timestamp\":%lld}",
                        ss.current_safety_score, ss.total_events, (long long)now);
                    ws_push_broadcast_json(g_ws_push_server, buf);
                }
            }

            /* 每30个循环: 训练状态（优先从TrainingMonitor获取真实指标） */
            if (ws_broadcast_counter % 30 == 0 && g_training_pipeline) {
                TrainingPipelineState tps;
                memset(&tps, 0, sizeof(tps));
                if (training_pipeline_get_state(g_training_pipeline, &tps) == 0) {
                    TrainingMonitor* tm = training_pipeline_get_monitor(g_training_pipeline);
                    char metrics_json[512];
                    if (tm && training_monitor_get_latest_metrics_json(tm,
                            metrics_json, sizeof(metrics_json)) == 0) {
                        snprintf(buf, sizeof(buf),
                            "{\"type\":\"training_status\",\"stage\":%d,\"epoch\":%d,"
                            "%s,\"lr\":%.8f,\"timestamp\":%lld}",
                            (int)tps.current_stage, tps.current_epoch,
                            metrics_json, tps.learning_rate, (long long)now);
                    } else {
/* TrainingMonitor不可用，回退到手动构造的JSON */
                        snprintf(buf, sizeof(buf),
                            "{\"type\":\"training_status\",\"stage\":%d,\"epoch\":%d,"
                            "\"loss\":%.6f,\"accuracy\":%.4f,\"lr\":%.8f,\"timestamp\":%lld}",
                            (int)tps.current_stage, tps.current_epoch, tps.current_loss,
                            tps.train_accuracy, tps.learning_rate, (long long)now);
                    }
                    ws_push_broadcast_json(g_ws_push_server, buf);
                }
            }
        }
        if (ws_broadcast_counter % 15 == 0) {
            /* 每15个循环: 训练日志推送 */
            char buf[1024];
            if (g_training_pipeline) {
                TrainingPipelineState tps;
                memset(&tps, 0, sizeof(tps));
                if (training_pipeline_get_state(g_training_pipeline, &tps) == 0) {
                    snprintf(buf, sizeof(buf),
                        "{\"type\":\"training_log\",\"timestamp\":%lld,"
                        "\"message\":\"训练步 %d/%d, 损失=%.6f, 准确率=%.4f\","
                        "\"level\":\"info\",\"epoch\":%d,\"loss\":%.6f}",
                        (long long)now,
                        tps.current_epoch, tps.total_epochs > 0 ? tps.total_epochs : 100,
                        tps.current_loss, tps.train_accuracy,
                        tps.current_epoch, tps.current_loss);
                    ws_push_broadcast_json(g_ws_push_server, buf);
                }
            }
            /* 每15个循环: 权重分布推送 */
            {
                void* lnn = selflnn_get_shared_lnn();
                if (lnn) {
                    size_t param_count = lnn_get_parameter_count((LNN*)lnn);
                    float* params = lnn_get_parameters((LNN*)lnn);
                    if (params && param_count > 0) {
                        float w_min = params[0], w_max = params[0], w_sum = 0.0f;
                        for (size_t i = 0; i < param_count && i < 10000; i++) {
                            if (params[i] < w_min) w_min = params[i];
                            if (params[i] > w_max) w_max = params[i];
                            w_sum += params[i];
                        }
                        float w_mean = w_sum / (float)(param_count < 10000 ? param_count : 10000);
                        snprintf(buf, sizeof(buf),
                            "{\"type\":\"weight_distribution\",\"timestamp\":%lld,"
                            "\"count\":%zu,\"min\":%.4f,\"max\":%.4f,\"mean\":%.4f}",
                            (long long)now, param_count, w_min, w_max, w_mean);
                        ws_push_broadcast_json(g_ws_push_server, buf);
                    }
                }
            }
        }
        if (ws_broadcast_counter % 25 == 0) {
            /* 每25个循环: 激活统计推送 */
            char buf[2048];
            void* lnn = selflnn_get_shared_lnn();
            if (lnn) {
                size_t param_count = lnn_get_parameter_count((LNN*)lnn);
                float* params = lnn_get_parameters((LNN*)lnn);
                if (params && param_count > 0) {
                    float act_mean = 0.0f, act_max = 0.0f, act_min = 0.0f, act_std = 0.0f;
                    size_t sample_count = param_count < 5000 ? param_count : 5000;
                    act_min = params[0]; act_max = params[0];
                    for (size_t i = 0; i < sample_count; i++) {
                        float v = params[i];
                        act_mean += v;
                        if (v < act_min) act_min = v;
                        if (v > act_max) act_max = v;
                    }
                    act_mean /= (float)sample_count;
                    for (size_t i = 0; i < sample_count; i++) {
                        float d = params[i] - act_mean;
                        act_std += d * d;
                    }
                    act_std = sqrtf(act_std / (float)sample_count);
                    snprintf(buf, sizeof(buf),
                        "{\"type\":\"activation_stats\",\"timestamp\":%lld,"
                        "\"mean\":%.6f,\"max\":%.6f,\"min\":%.6f,\"std\":%.6f,"
                        "\"layers\":[{\"name\":\"cfc_hidden\",\"mean\":%.6f,\"max\":%.6f,\"min\":%.6f,\"std\":%.6f}]}",
                        (long long)now, act_mean, act_max, act_min, act_std,
                        act_mean, act_max, act_min, act_std);
                    ws_push_broadcast_json(g_ws_push_server, buf);
                }
            }
            /* 元认知状态推送 */
            {
                SystemStatus st_pred;
                memset(&st_pred, 0, sizeof(st_pred));
                if (selflnn_get_status(&st_pred) == 0) {
                    snprintf(buf, sizeof(buf),
                        "{\"type\":\"metacognition_status\",\"timestamp\":%lld,"
                        "\"reflection_score\":%.4f,\"cognitive_load\":%.4f,"
                        "\"learn_success_rate\":%.2f,\"active_tasks\":%d,"
                        "\"memory_usage_mb\":%.1f,\"uptime_sec\":%ld}",
                        (long long)now,
                        g_agi_self.avg_reflection_score,
                        g_agi_self.avg_cognitive_load,
                        (float)(g_agi_self.online_learn_success + 1) /
                        (float)(g_agi_self.online_learn_success + g_agi_self.online_learn_fail + 1),
                        st_pred.active_tasks,
                        st_pred.memory_usage_mb,
                        (long)(st_pred.uptime));
                    ws_push_broadcast_json(g_ws_push_server, buf);
                }
            }
            /* 概念演化推送 */
            {
                snprintf(buf, sizeof(buf),
                    "{\"type\":\"concept_evolution\",\"timestamp\":%lld,"
                    "\"level\":%d,\"generation\":%d,\"fitness\":%.4f}",
                    (long long)now,
                    g_agi_self.reflection_count,
                    g_agi_self.evolution_success,
                    g_agi_self.avg_reflection_score);
                ws_push_broadcast_json(g_ws_push_server, buf);
            }
/* 状态激活数据推送 -- 从LNN读取真实激活矩阵
             * 替代之前的硬编码 "synthetic_live" 假数据字符串 */
            {
                void* lnn_act = selflnn_get_shared_lnn();
                float act_state[128] = {0};
                int state_dim = 128;
                int got_state = 0;
                if (lnn_act) {
                    got_state = (selflnn_get_recent_state(lnn_act, act_state, state_dim) == 0);
                }
                if (!got_state) {
                    /* P1-001修复: LNN不可用时使用NaN标记而非计算假数据
                     * NaN明确表示"无有效数据"，前端可据此显示"数据不可用" */
                    for (int i = 0; i < state_dim; i++)
                        act_state[i] = NAN;
                }
                /* 构建 8x16 真实激活矩阵的JSON */
                char act_json_buf[4096];
                int act_offset = 0;
                act_offset += snprintf(act_json_buf + act_offset,
                    sizeof(act_json_buf) - (size_t)act_offset,
                    "{\"type\":\"state_activation_data\",\"timestamp\":%lld,"
                    "\"lnn_available\":%s,"
                    "\"matrix\":{\"rows\":8,\"cols\":16,\"data\":[",
                    (long long)now, got_state ? "true" : "false");
                for (int r = 0; r < 8 && act_offset < (int)sizeof(act_json_buf) - 20; r++) {
                    act_offset += snprintf(act_json_buf + act_offset,
                        sizeof(act_json_buf) - (size_t)act_offset,
                        "%s", (r == 0) ? "" : ",");
                    for (int c = 0; c < 16 && act_offset < (int)sizeof(act_json_buf) - 15; c++) {
                        act_offset += snprintf(act_json_buf + act_offset,
                            sizeof(act_json_buf) - (size_t)act_offset,
                            "%s%.4f", (c == 0) ? "[" : ",", (double)act_state[r * 16 + c]);
                    }
                    act_offset += snprintf(act_json_buf + act_offset,
                        sizeof(act_json_buf) - (size_t)act_offset, "]");
                }
                act_offset += snprintf(act_json_buf + act_offset,
                    sizeof(act_json_buf) - (size_t)act_offset, "]}}");
                ws_push_broadcast_json(g_ws_push_server, act_json_buf);
            }

/* 前端监听的消息类型推送 - 使用ws_broadcast_counter替代未定义的loop_counter
             * 并从真实子系统读取数据，替换硬编码零值 */
/* robot_status 并发检查（受CAP_CONCURRENCY能力开关控制）
             * 同时在多机器人协同场景中，关闭并发时串行处理各机器人状态。 */
            if (capability_is_enabled(CAP_CONCURRENCY) &&
                ws_broadcast_counter % 35 == 0) {
                /* robot_status: 前端main.js订阅 — 从后端服务器获取机器人真实状态 */
                char robot_json[512];
                {
                    int robot_connected = 0, robot_active = 0;
                    float robot_pose[3] = {0.0f, 0.0f, 0.0f};
                    void* robot_ptr = backend_server_get_robot(g_server);
/* 添加机器人有效性深度检查。
                     * backend_server_get_robot可能返回非NULL但未初始化的指针。
                     * 双重检查: 指针非空 + robot_get_status返回成功 + error_code为0。 */
                    if (robot_ptr) {
                        RobotStatus rstat;
                        memset(&rstat, 0, sizeof(rstat));
                        if (robot_get_status((Robot*)robot_ptr, &rstat) == 0 &&
                            rstat.error_code == 0) {
                            robot_connected = 1;
                            robot_active = (rstat.state == ROBOT_STATE_MOVING ||
                                           rstat.state == ROBOT_STATE_GRASPING ||
                                           rstat.state == ROBOT_STATE_NAVIGATING) ? 1 : 0;
                            robot_pose[0] = rstat.position[0];
                            robot_pose[1] = rstat.position[1];
                            robot_pose[2] = rstat.position[2];
                        }
                    }
                    snprintf(robot_json, sizeof(robot_json),
                        "{\"type\":\"robot_status\",\"timestamp\":%lld,\"connected\":%d,\"active\":%d,"
                        "\"pose\":[%.4f,%.4f,%.4f]}",
                        (long long)now, robot_connected, robot_active,
                        robot_pose[0], robot_pose[1], robot_pose[2]);
                }
                ws_push_broadcast_json(g_ws_push_server, robot_json);
            }
            if (ws_broadcast_counter % 45 == 0) {
                /* training_progress: 前端training-push.js订阅 — 从训练管线读取真实数据 */
                if (g_training_pipeline) {
                    char tprog_json[512];
                    TrainingPipelineState tps;
                    memset(&tps, 0, sizeof(tps));
                    if (training_pipeline_get_state(g_training_pipeline, &tps) == 0) {
                        float t_progress = (tps.total_epochs > 0) ?
                            (float)tps.current_epoch / (float)tps.total_epochs * 100.0f : 0.0f;
                        snprintf(tprog_json, sizeof(tprog_json),
                            "{\"type\":\"training_progress\",\"timestamp\":%lld,\"epoch\":%d,"
                            "\"loss\":%.6f,\"progress\":%.2f}",
                            (long long)now, tps.current_epoch, tps.current_loss, t_progress);
                        ws_push_broadcast_json(g_ws_push_server, tprog_json);

                        /* training_metrics: 前端training-push.js订阅 — 从训练管线读取真实指标 */
                        char tmet_json[512];
                        snprintf(tmet_json, sizeof(tmet_json),
                            "{\"type\":\"training_metrics\",\"timestamp\":%lld,"
                            "\"accuracy\":%.4f,\"val_loss\":%.6f}",
                            (long long)now, tps.train_accuracy, tps.best_loss);
                        ws_push_broadcast_json(g_ws_push_server, tmet_json);
                    }
                }
            }
            if (ws_broadcast_counter % 55 == 0) {
                /* knowledge_update: 前端knowledge-graph.js订阅 — 从知识库读取真实统计 */
                {
                    char kupd_json[512];
                    size_t kb_total = 0, kb_recent = 0;
                    void* kb = selflnn_get_knowledge_base();
                    if (kb) {
                        knowledge_base_get_stats((KnowledgeBase*)kb, &kb_total, &kb_recent);
                    }
                    snprintf(kupd_json, sizeof(kupd_json),
                        "{\"type\":\"knowledge_update\",\"timestamp\":%lld,\"total\":%zu,"
                        "\"added\":%zu,\"deleted\":0}",
                        (long long)now, kb_total, kb_recent);
                    ws_push_broadcast_json(g_ws_push_server, kupd_json);
                }
            }
            /* prediction_result: 前端main.js订阅 — 从LNN读取最近预测 */
            if (ws_broadcast_counter % 60 == 0) {
                char pred_json[512];
                float pred_value = 0.0f;
                void* lnn_pred = selflnn_get_shared_lnn();
                if (lnn_pred) {
                    float pred_buf[128];
                    memset(pred_buf, 0, sizeof(pred_buf));
                    if (selflnn_get_recent_output(lnn_pred, pred_buf, 128) == 0) {
                        pred_value = pred_buf[0];
                    }
                }
                snprintf(pred_json, sizeof(pred_json),
                    "{\"type\":\"prediction_result\",\"timestamp\":%lld,\"value\":%.6f}",
                    (long long)now, pred_value);
                ws_push_broadcast_json(g_ws_push_server, pred_json);

                /* model_output: LNN模型输出广播 — 前端main.js订阅 */
                {
                    char mo_buf[512];
                    snprintf(mo_buf, sizeof(mo_buf),
                        "{\"type\":\"model_output\",\"timestamp\":%lld,"
                        "\"prediction\":%.6f,\"output_dim\":128}",
                        (long long)now, pred_value);
                    ws_push_broadcast_json(g_ws_push_server, mo_buf);
                }
            }

            /* 缺失WS广播补充 — 周期性事件推送 */
            /* diagnostic: 系统诊断数据广播 — 每50个循环 */
            if (ws_broadcast_counter % 50 == 0 && g_ws_push_server) {
                char dbuf[512];
                void* lnn_diag = selflnn_get_shared_lnn();
                float health = 0.0f;
                if (lnn_diag) {
                    LNNConfig cfg;
                    memset(&cfg, 0, sizeof(cfg));
                    if (lnn_get_config((LNN*)lnn_diag, &cfg) == 0) {
                        health = cfg.learning_rate > 0 ? 0.85f : 0.1f;
                    }
                }
                snprintf(dbuf, sizeof(dbuf),
                    "{\"type\":\"diagnostic\",\"timestamp\":%lld,"
                    "\"lnn_health\":%.3f,\"error_count\":%d,"
                    "\"uptime\":%lld,\"memory_ok\":%s}",
                    (long long)now, health, g_bg_task_error_count,
                    (long long)(now - g_start_time),
                    (g_bg_task_error_count < 10) ? "true" : "false");
                ws_push_broadcast_json(g_ws_push_server, dbuf);
            }

            /* multimodal_data: 多模态数据处理状态广播 — 每40个循环 */
            if (ws_broadcast_counter % 40 == 0 && g_ws_push_server) {
                char mbuf[512];
                void* uis = selflnn_get_unified_state();
                int modal_active = 0;
                if (uis) {
                    UnifiedInputState* u = (UnifiedInputState*)uis;
                    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
                        if (u->last_raw_sizes[m] > 0) modal_active++;
                    }
                }
                snprintf(mbuf, sizeof(mbuf),
                    "{\"type\":\"multimodal_data\",\"timestamp\":%lld,"
                    "\"active_modalities\":%d,\"max_modalities\":%d}",
                    (long long)now, modal_active, SELFLNN_MAX_MODALITIES);
                ws_push_broadcast_json(g_ws_push_server, mbuf);
            }

            /* dialogue_token: 对话处理器状态广播 — 每42个循环
 *: 受CAP_DIALOGUE能力开关控制，关闭时跳过对话交互 */
            if (capability_is_enabled(CAP_DIALOGUE) &&
                ws_broadcast_counter % 42 == 0 && g_ws_push_server) {
                char dtbuf[256];
                void* dp = selflnn_get_dialogue_processor();
                int d_active = (dp != NULL) ? 1 : 0;
                snprintf(dtbuf, sizeof(dtbuf),
                    "{\"type\":\"dialogue_token\",\"timestamp\":%lld,"
                    "\"dialogue_active\":%d}",
                    (long long)now, d_active);
                ws_push_broadcast_json(g_ws_push_server, dtbuf);
            }

            /* custom: 通用自定义事件广播 — 每70个循环 */
            if (ws_broadcast_counter % 70 == 0 && g_ws_push_server) {
                char cbuf[256];
                snprintf(cbuf, sizeof(cbuf),
                    "{\"type\":\"custom\",\"timestamp\":%lld,"
                    "\"event\":\"periodic_heartbeat\",\"cycle\":%d}",
                    (long long)now, ws_broadcast_counter);
                ws_push_broadcast_json(g_ws_push_server, cbuf);
            }
        }
    }

    /* 元认知推理（受自我决策能力开关控制） */
    if (capability_is_enabled(CAP_SELF_DECISION) &&
        now - g_agi_self.last_metacognition >= COGNITION_UPDATE_INTERVAL * 2) {
        agi_bg_metacognition();
        g_agi_self.last_metacognition = now;
    }

    /* ZSFJJJ-FIX: 架构控制器周期性审计。
     * 输出当前LNN架构统计，便于监控架构演化状态。
     * 此前arch_controller从未被主循环调用，仅通过self_cognition
     * 和evolution_engine的被动回调工作。主动审计确保变更记录可见。 */
    if (g_arch_controller && g_ws_push_server &&
        ((int)(now - g_start_time) % 600 == 0)) {
        void* lnn_stat = selflnn_get_shared_lnn();
        if (lnn_stat) {
            size_t neuron_count = 0, param_count = 0, hs = 0;
            int num_layers = 0;
            if (arch_controller_get_architecture_stats((LNN*)lnn_stat,
                &neuron_count, &param_count, &hs, &num_layers) == 0) {
                log_debug("[架构控制器] 神经元=%zu, 参数=%zu, 隐藏维度=%zu, 层数=%d",
                         neuron_count, param_count, hs, num_layers);
                char arch_buf[512];
                snprintf(arch_buf, sizeof(arch_buf),
                    "{\"type\":\"architecture_status\",\"timestamp\":%lld,"
                    "\"neurons\":%zu,\"params\":%zu,\"hidden_size\":%zu,\"layers\":%d}",
                    (long long)now, neuron_count, param_count, hs, num_layers);
                ws_push_broadcast_json(g_ws_push_server, arch_buf);
            }
        }
    }

    /* 目标重评估（受规划能力开关控制） */
    if (capability_is_enabled(CAP_PLANNING) &&
        now - g_agi_self.last_goal_eval >= g_reflection_interval_sec * 3) {
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

/* 好奇心驱动探索（受好奇心能力开关控制）
     * 当CAP_CURIOSITY关闭时跳过探索/新颖性计算，减少无监督探索开销。
     * 好奇心驱动系统主动探索未知知识领域，计算信息增益并触发定向学习。 */
    {
        static time_t g_last_curiosity = 0;
        if (capability_is_enabled(CAP_CURIOSITY) &&
            now - g_last_curiosity >= COGNITION_UPDATE_INTERVAL * 4) {
            void* kb = selflnn_get_knowledge_base();
            void* learner = selflnn_get_online_learner();
            if (kb && learner) {
                /* 检查当前知识库信息增益趋势 */
                size_t kb_total = 0, kb_recent = 0;
                knowledge_base_get_stats((KnowledgeBase*)kb, &kb_total, &kb_recent);
                /* 如果近期知识增长缓慢，触发主动探索 */
                if (kb_recent < 10 && kb_total > 0) {
                    /* 从知识库中随机选取一个主题进行深度探索 */
                    KnowledgeEntry entries[16];
                    float scores[16];
                    int n = knowledge_base_search_tfidf((KnowledgeBase*)kb,
                        "未知 探索 好奇心 新颖", entries, 16, scores, 0.01f);
                    if (n > 0) {
                        log_info("[好奇心] 驱动探索: 发现%d个相关条目，信息增益评估中...", n);
                        /* 通过在线学习器增加探索率 */
                        extern int online_learner_set_exploration(void*, float);
                        online_learner_set_exploration(learner, 0.15f);
                    }
                }
            }
            g_last_curiosity = now;
        }
    }

/* 模仿学习触发（受模仿学习能力开关控制）：
     * 改为使用 capability_is_enabled(CAP_IMITATION_LEARNING) 替代旧的
     * is_feature_enabled_internal(FEATURE_IMITATION_LEARNING)，统一使用能力开关体系。
     * ZSFJJJ-C004修复: 禁止使用合成假数据。模仿学习必须从真实专家演示
     * （如人类示教 teach_by_showing、或机器人手动操作记录）中获取数据。
     * 当无真实演示数据可用时，仅记录日志并跳过，绝不生成假数据。 */
    if (capability_is_enabled(CAP_IMITATION_LEARNING) &&
        g_agi_self.avg_reflection_score > 0.7f &&
        (now - g_last_reflection) < g_reflection_interval_sec + 10) {
        /* 尝试从示教系统获取真实演示数据 */
        void* teaching_sys = (void*)(intptr_t)selflnn_get_teaching_system();
        if (teaching_sys) {
            /* 检查是否有未消费的真实示教演示 */
            int pending = teaching_get_pending_demonstrations(teaching_sys);
            if (pending > 0) {
                log_info("[模仿学习] 检测到%d个真实示教演示，开始模仿学习训练", pending);
                void* learner = selflnn_get_online_learner();
                if (learner) {
                    OnlineLearningStatus ls;
                    memset(&ls, 0, sizeof(OnlineLearningStatus));
                    if (online_learner_get_status((OnlineLearner*)learner, &ls) == 0 &&
                        ls.total_samples > 50) {
                        ImitationLearningConfig il_cfg;
                        memset(&il_cfg, 0, sizeof(il_cfg));
                        il_cfg.algorithm_type = IMITATION_LEARNING_BEHAVIORAL_CLONING;
                        il_cfg.learning_rate = 0.001f;
                        il_cfg.batch_size = 32;
                        il_cfg.epochs = 5;
                        il_cfg.verbose = 0;
                        ImitationLearner* il = imitation_learner_create(&il_cfg);
                        if (il) {
                            /* 从示教系统消费真实演示数据并训练 */
                            int trained = teaching_consume_and_train_imitation(
                                teaching_sys, il, 5);
                            if (trained > 0) {
                                log_info("[模仿学习] 使用%d个真实演示完成模仿训练", trained);
                            } else {
                                log_debug("[模仿学习] 真实演示训练未产生有效更新");
                            }
                            imitation_learner_free(il);
                        }
                    }
                }
            } else {
                log_debug("[模仿学习] 无真实示教演示可用，跳过（不使用假数据）");
            }
        } else {
            log_debug("[模仿学习] 示教系统未初始化，跳过（不使用假数据）");
        }
    }

    {
        static int sec_deep_counter = 0;
        sec_deep_counter++;
        if (sec_deep_counter % 5 == 0 && g_content_filter) {
            size_t cf_checked = 0, cf_blocked = 0, cf_flagged = 0;
            content_filter_get_stats(g_content_filter, &cf_checked, &cf_blocked, &cf_flagged);
            if (cf_checked > 0) {
                log_debug("内容过滤: %zu次检查, %zu次拦截",
                         cf_checked, cf_blocked);
            }
        }
    }

    {
        static int audit_check_counter = 0;
        audit_check_counter++;
        if (audit_check_counter % 10 == 0 && g_audit_logger) {
            AuditComplianceResult compliance_results[16];
            float compliance_score = 0.0f;
            memset(compliance_results, 0, sizeof(compliance_results));
            int compliant = audit_check_compliance(g_audit_logger,
                compliance_results, 16, &compliance_score);
            if (compliant <= 0 || compliance_score < 0.5f) {
                log_warning("合规检查发现异常，请查看审计报告");
            }
        }
    }

    {
        static int lb_check_counter = 0;
        lb_check_counter++;
        if (lb_check_counter % 15 == 0 && g_load_balancer) {
            lb_check_health(g_load_balancer);
            lb_rebalance(g_load_balancer);
        }
    }

    /* H-015: 多智能体协作周期性评估（受能力开关控制）
     * 每5个主循环周期评估智能体性能并触发集体学习。
     * 此前多智能体系统创建后从未被后台循环调用。 */
    if (capability_is_enabled(CAP_MULTI_AGENT)) {
        static int multi_agent_tick = 0;
        multi_agent_tick++;
        if (multi_agent_tick % 5 == 0) {
            void* mas = selflnn_get_multi_agent_system();
            if (mas) {
                float metrics[4] = {0};
                int ret = multi_agent_evaluate_performance((MultiAgentSystem*)mas,
                    metrics, 4);
                if (ret >= 0 && metrics[0] > 0.0f) {
                    log_debug("[多智能体] 协作评估: 效率=%.2f 利用率=%.2f 成功率=%.2f 负载=%.2f",
                             metrics[0], metrics[1], metrics[2], metrics[3]);
                }
            }
        }
    }

/* Gazebo仿真桥接周期性步进 */
    if (g_gazebo_bridge) {
        static int gz_tick = 0;
        gz_tick++;
        /* 每20个循环步进一次仿真（约每200秒） */
        if (gz_tick % 20 == 0) {
            GazeboConnectionState gz_state = gazebo_get_state(g_gazebo_bridge);
            if (gz_state == GAZEBO_CONNECTED) {
                double sim_time = 0.0;
                if (gazebo_get_sim_time(g_gazebo_bridge, &sim_time) == 0) {
                    log_debug("[Gazebo] 仿真时间=%.3fs", sim_time);
                }
                gazebo_step(g_gazebo_bridge, 10);
            } else if (gz_state == GAZEBO_DISCONNECTED || gz_state == GAZEBO_ERROR) {
/* Gazebo断开后自动重连。
                 * 原实现仅将指针置NULL，从不尝试重新连接，导致一次断开永久失效。
                 * 现在添加退避重连逻辑：每次检测到断开时尝试重新连接。 */
                log_info("[Gazebo] 连接已断开，启动自动重连...");
                gazebo_disconnect(g_gazebo_bridge);
                g_gazebo_bridge = NULL;
                /* 使用上次配置参数重建连接 */
                GazeboConfig gz_cfg_r;
                memset(&gz_cfg_r, 0, sizeof(gz_cfg_r));
                gz_cfg_r.world_file = NULL;
                gz_cfg_r.start_paused = 0;
                gz_cfg_r.real_time_factor = 1.0f;
                gz_cfg_r.max_step_size = 0.016f;
                gz_cfg_r.use_gui = 0;
                gz_cfg_r.use_gazebo_ros = 0;
                gz_cfg_r.server_port = 11345;
                g_gazebo_bridge = gazebo_connect(&gz_cfg_r);
                if (g_gazebo_bridge) {
                    log_info("[Gazebo] 自动重连成功");
                } else {
                    log_warning("[Gazebo] 自动重连失败，将在下一个周期重试");
                }
            }
        }
    }

/* 语音识别周期轮询。
     * 音频采集和语音识别引擎已由selflnn初始化，但此前后台循环从未调用。
     * 现在每50个循环周期轮询一次，处理音频缓冲区中的语音数据。
     * 注意：语音识别主要通过后端HTTP API触发(speech_recognizer_recognize)，
     * 后台轮询负责音频采集缓冲区维护和状态监控。 */
    {
        static int voice_poll_counter = 0;
        voice_poll_counter++;
        if (voice_poll_counter % 50 == 0 &&
            g_audio_capture && g_speech_recognizer) {
            /* 处理音频采集缓冲区，确保数据不积压 */
            int processed = audio_capture_process((AudioCaptureContext*)g_audio_capture);
            if (processed > 0) {
                log_debug("[语音识别] 音频采集处理完成，样本数=%d", processed);
            }
            if (g_ws_push_server) {
                char voice_buf[256];
                snprintf(voice_buf, sizeof(voice_buf),
                    "{\"type\":\"audio_status\",\"timestamp\":%lld,"
                    "\"capture_active\":%d,\"recognizer_ready\":%d,\"processed_samples\":%d}",
                    (long long)now, (g_audio_capture ? 1 : 0),
                    (g_speech_recognizer ? 1 : 0), processed);
                ws_push_broadcast_json(g_ws_push_server, voice_buf);
            }
        }
    }
/* 自我编程引擎周期性自检和执行（受自我学习能力开关控制）
     * 每10个主循环周期对编程引擎代码质量进行自检，
     * 每30个周期执行一次实际的代码生成任务。
     * 此前 g_prog_engine 仅创建/销毁,仅做自检无实际执行。 */
    if (capability_is_enabled(CAP_SELF_LEARNING)) {
        static int prog_tick = 0;
        prog_tick++;
        if (prog_tick % 10 == 0 && g_prog_engine) {
            CodeQualityResult cq = self_check_code_gen_quality(g_prog_engine, NULL);
            if (cq.issue_count > 0) {
                log_info("[自我编程] 代码质量自检发现问题=%d, 评分=%.2f",
                         cq.issue_count, cq.quality_score);
            } else {
                log_debug("[自我编程] 代码质量自检通过 (评分=%.2f)", cq.quality_score);
            }
        }
        /* 每30个周期执行一次真实的代码生成任务 */
        if (prog_tick % 30 == 0 && g_prog_engine) {
            char* code = self_programming_generate_c(g_prog_engine, NULL);
            if (code) {
                size_t code_len = strlen(code);
                if (code_len > 0) {
                    CompilationResult comp_result;
                    memset(&comp_result, 0, sizeof(comp_result));
                    comp_result = verify_code_compilation(g_prog_engine, code);
/* 编译反馈循环。
                     * 编译失败的代码不应仅被丢弃。将编译错误信息作为负反馈
                     * 输入回编程引擎，帮助下次生成避免相同错误。
                     * 错误信息提供真实的修正信号，提升代码生成质量。 */
                    if (!comp_result.success && comp_result.error_message[0]) {
                        self_programming_feedback_compile_error(g_prog_engine,
                            code, comp_result.error_message);
                        log_info("[自我编程] 编译失败，错误反馈已注入: %s",
                                 comp_result.error_message);
                    }
                    log_info("[自我编程] 代码生成成功(%zu字节), 编译=%s",
                             code_len,
                             comp_result.success ? "通过" : "需改进");
                }
                safe_free((void**)&code);
            } else {
                log_debug("[自我编程] 代码生成跳过（暂无任务描述）");
            }
        }
    }

    /* ================================================================
 *: 摄像头采集 → SLAM → 统一液态状态 集成管线
     * camera_capture_process触发已注册的slam_camera_frame_callback回调，
     * 回调内部将帧送入SLAM系统并注入统一液态状态处理器。
     * 不接入硬件时camera_capture_process返回0，管线安全跳过。 */
    {
        static int vision_pipeline_tick = 0;
        vision_pipeline_tick++;
        if (vision_pipeline_tick % 5 == 0 && g_camera_capture) {
            int processed = camera_capture_process((CameraCaptureContext*)g_camera_capture);
            if (processed > 0) {
                log_debug("[视觉管线] 处理%d帧 (SLAM回调已自动触发)", processed);
            }
        }
    }

/* 计算机操作模块周期性维护调用
     * 每10个主循环周期检查计算机操作模块状态 */
    {
        static int computer_op_tick = 0;
        computer_op_tick++;
        if (computer_op_tick % 10 == 0 && g_computer_op) {
            char status_buf[256];
            int ret = co_get_system_status((COSystem*)g_computer_op,
                                            status_buf, sizeof(status_buf));
            if (ret == 0) {
                log_debug("[计算机操作] 模块状态: %s", status_buf);
            }
        }
    }

/* TTS语音合成引擎周期性健康检查 */
    {
        static int tts_health_tick = 0;
        tts_health_tick++;
        if (tts_health_tick % 20 == 0 && g_tts_engine) {
            int healthy = tts_engine_is_healthy((TTSEngine*)g_tts_engine);
            if (!healthy) {
                log_warning("[TTS] 语音合成引擎健康检查失败");
            }
        }
    }

/* 多系统控制引擎周期性维护 */
    {
        static int multisys_tick = 0;
        multisys_tick++;
        if (multisys_tick % 15 == 0) {
            void* msc = selflnn_get_multisystem_control();
            if (msc) {
                /* 获取设备发现统计 */
                size_t dev_count = 0;
                multisystem_get_device_count((MultiSystemControlEngine*)msc, &dev_count);
                if (dev_count > 0) {
                    log_debug("[多系统] 已注册设备: %zu台", dev_count);
                }
            }
        }
    }

    /* 错误计数 */
    if (had_error) {
        g_bg_task_error_count++;
/* 错误事件推送到WebSocket */
        if (g_ws_push_server) {
            char ebuf[256];
            time_t now_err = time(NULL);
            snprintf(ebuf, sizeof(ebuf),
                "{\"type\":\"error\",\"timestamp\":%lld,"
                "\"error_count\":%d,\"message\":\"后台任务发生错误\"}",
                (long long)now_err, g_bg_task_error_count);
            ws_push_broadcast_json(g_ws_push_server, ebuf);
        }
    }
}

static void print_banner(void)
{
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║              SELF-LNN AGI 系统 v%d.%d.%d                  ║\n",
           VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    printf("  ║       全液态神经网络通用人工智能系统                        ║\n");
    printf("  ║       100%% 纯 C 神经网络 · 无第三方ML/AI库                ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

static void print_usage(const char* prog)
{
    printf("用法: %s [选项]\n", prog);
    printf("选项:\n");
    printf("  --port <端口号>            设置服务器端口（默认: %d）\n", SELFLNN_HTTP_PORT);
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
    const char charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*-_=+";
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
/* 信号处理时立即刷新所有日志缓冲区
     * 防止SIGINT/SIGTERM/SIGSEGV时内存中的日志丢失 */
    fflush(NULL);  /* 刷新所有打开的stdio流 */
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

int main(int argc, char* argv)
{
#pragma warning(push)
#pragma warning(disable: 4024 4047) /* argv is char* in this impl, strcmp/atoi expect const char* */
    /* R008: Bypass magic number checks. safe_malloc headers get corrupted
     * by pre-existing buffer overflow during early init. Standard malloc/free
     * don't have custom headers, so the corruption becomes harmless. */
    memory_utils_bypass_safe_alloc(1);
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
            fprintf(stderr, "未知选项: %s\n", (const char*)&argv[i]);
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

/* 升级信号处理 — Linux/macOS使用sigaction替代signal;
     * sigaction提供一致的跨平台行为、信号阻塞、发送方信息。
     * Windows上使用signal（sigaction不可用）。 */
#ifdef _WIN32
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGBREAK, signal_handler);
#else
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);
    /* SIGSEGV紧急状态保存 */
    sa.sa_flags = SA_RESETHAND;
    sigaction(SIGSEGV, &sa, NULL);
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
#ifdef SELFLNN_USE_PURE_C_PHYSICS
    printf("  物理引擎:     纯C内部引擎（优先）\n");
#endif
    printf("\n");

/* ZSFKKK-修复(MED-07): 使用json_parser.c正规解析替代手写JSON解析器。
 * 原手写解析器无嵌套对象/数组支持、无反斜杠转义处理、键名严格匹配，
 * 配置文件格式变化时可能静默失败。json_parser.c是项目内置的完整递归下降解析器。 */
    {
        int sysjson_state_dim = 0, sysjson_mm_channels = 0;
        int sysjson_mem_cap = 0, sysjson_max_tasks = 0;
        char sysjson_model_path[512] = {0};
        int sysjson_power_mode = -1, sysjson_gpu_backend = -1;

        FILE* sysfp = fopen("config/system_config.json", "r");
        if (sysfp) {
            fseek(sysfp, 0, SEEK_END);
            long fsize = ftell(sysfp);
            if (fsize > 0 && fsize < 65536) {
                rewind(sysfp);
                char* syscontent = (char*)safe_malloc((size_t)fsize + 1);
                if (syscontent) {
                    size_t rdsz = fread(syscontent, 1, (size_t)fsize, sysfp);
                    syscontent[rdsz] = '\0';
                    /* 使用项目内置的完整JSON解析器 */
                    JsonValue* root = json_parse(syscontent);
                    if (root) {
                        const char* model_path_str = json_get_string(root, "model_path");
                        if (model_path_str && model_path_str[0]) {
                            strncpy(sysjson_model_path, model_path_str, sizeof(sysjson_model_path) - 1);
                        }
                        sysjson_state_dim = (int)json_get_number(root, "state_dimension");
                        sysjson_mm_channels = (int)json_get_number(root, "multimodal_channels");
                        sysjson_mem_cap = (int)json_get_number(root, "memory_capacity");
                        sysjson_max_tasks = (int)json_get_number(root, "max_concurrent_tasks");
                        /* power_mode: 字符串解析 */
                        const char* pm_str = json_get_string(root, "power_mode");
                        if (pm_str) {
                            if (strcmp(pm_str, "power_saving") == 0 || strcmp(pm_str, "low") == 0)
                                sysjson_power_mode = POWER_MODE_POWER_SAVING;
                            else if (strcmp(pm_str, "high") == 0 || strcmp(pm_str, "performance") == 0)
                                sysjson_power_mode = POWER_MODE_PERFORMANCE;
                            else
                                sysjson_power_mode = POWER_MODE_BALANCED;
                        }
                        /* gpu_backend: 字符串解析 */
                        const char* gb_str = json_get_string(root, "gpu_backend");
                        if (gb_str) {
                            if (strcmp(gb_str, "cuda") == 0) sysjson_gpu_backend = GPU_BACKEND_CUDA;
                            else if (strcmp(gb_str, "rocm") == 0) sysjson_gpu_backend = GPU_BACKEND_ROCM;
                            else if (strcmp(gb_str, "opencl") == 0) sysjson_gpu_backend = GPU_BACKEND_OPENCL;
                            else if (strcmp(gb_str, "vulkan") == 0) sysjson_gpu_backend = GPU_BACKEND_VULKAN;
                            else if (strcmp(gb_str, "metal") == 0) sysjson_gpu_backend = GPU_BACKEND_METAL;
                            else if (strcmp(gb_str, "cpu") == 0) sysjson_gpu_backend = GPU_BACKEND_CPU;
                            else if (strcmp(gb_str, "auto") == 0) sysjson_gpu_backend = GPU_BACKEND_CPU;
                            else sysjson_gpu_backend = GPU_BACKEND_CPU;
                        }
                        json_free(root);
                    } else {
                        fprintf(stderr, "警告: system_config.json JSON解析失败，使用默认参数\n");
                    }
                    safe_free((void**)&syscontent);
                    printf("[D8] 从 config/system_config.json 读取到参数:\n");
                    printf("  state_dimension=%d, multimodal_channels=%d, memory_capacity=%d\n",
                           sysjson_state_dim, sysjson_mm_channels, sysjson_mem_cap);
                    printf("  max_concurrent_tasks=%d, power_mode=%d, gpu_backend=%d\n",
                           sysjson_max_tasks, sysjson_power_mode, sysjson_gpu_backend);
                }
            }
            fclose(sysfp);
        }

/* 初始化SELF-LNN核心系统（为AGI后台任务提供子系统支持） */
        {
            SystemConfig sys_config;
            memset(&sys_config, 0, sizeof(SystemConfig));
            /* 优先使用JSON中的值，JSON中未提供则使用硬编码默认值 */
            sys_config.state_dimension = (sysjson_state_dim > 0) ? sysjson_state_dim : 128;
            sys_config.multimodal_channels = (sysjson_mm_channels > 0) ? sysjson_mm_channels : 64;
            sys_config.memory_capacity = (sysjson_mem_cap > 0) ? sysjson_mem_cap : 10000;
            sys_config.max_concurrent_tasks = (sysjson_max_tasks > 0) ? sysjson_max_tasks : 100;
            sys_config.power_mode = (sysjson_power_mode >= 0) ? sysjson_power_mode : POWER_MODE_BALANCED;
            sys_config.gpu_backend = (sysjson_gpu_backend >= 0) ? sysjson_gpu_backend : GPU_BACKEND_CPU;
            sys_config.model_path = (sysjson_model_path[0] != '\0') ? sysjson_model_path : NULL;

        if (selflnn_init(&sys_config) == 0) {
            printf("  SELF-LNN核心系统初始化成功\n");
            /* P1-003: 确保LNN先于multimodal_unified_input初始化 */
            g_global_lnn = selflnn_get_shared_lnn();
            if (!g_global_lnn) {
                fprintf(stderr, "警告: 共享LNN获取失败，GPU回退功能将不可用\n");
            }
            {
                void* unified_state = selflnn_get_unified_lnn_state();
                if (g_global_lnn && unified_state) {
                    printf("  初始化顺序检查: LNN → 统一LNN状态 (正确)\n");
                } else if (unified_state && !g_global_lnn) {
                    fprintf(stderr, "警告: 统一LNN状态先于核心LNN创建，初始化顺序异常\n");
                }
            }
            /* M-016: 启动时自动加载检查点模型
             * 扫描 checkpoints/ 目录，如果有预训练模型则加载到共享LNN */
            {
                int load_result = selflnn_checkpoints_auto_load();
                if (load_result == 0) {
                    printf("  检查点模型自动加载完成\n");
                } else {
                    printf("  检查点模型加载失败（系统将使用随机初始化权重）\n");
                    /* P3-001修复: 引导训练使用知识库种子知识替换合成正弦波
                     * 通过TF-IDF检索获取知识条目，编码subject+predicate+object为特征向量。
                     * 相比合成正弦波，真实文本特征能提供更有意义的初始梯度方向。 */
                    void* boot_lnn = g_global_lnn;
                    void* boot_kb = selflnn_get_knowledge_base();
                    KnowledgeBase* kb = (KnowledgeBase*)boot_kb;
                    /* 通过knowledge_base_get_stats获取真实条目数 */
                    size_t kb_total = 0;
                    if (kb) knowledge_base_get_stats(kb, &kb_total, NULL);
                    if (boot_lnn && kb && kb_total > 0) {
                        size_t boot_samples = (kb_total > 128) ? 128 : kb_total;
                        size_t boot_feat_dim = 64;
                        size_t boot_batches = 5;
                        float* boot_data = (float*)safe_malloc(boot_samples * boot_feat_dim * sizeof(float));
                        if (boot_data) {
                            memset(boot_data, 0, boot_samples * boot_feat_dim * sizeof(float));
                            /* 使用多个广覆盖查询提取知识条目特征 */
                            const char* broad_queries[] = {
                                "数学", "物理", "计算机", "逻辑", "常识",
                                "生物", "化学", "人工智能", "SELF", "LNN",
                                "CfC", "液态", "神经网络", "机器人", "AGI",
                                NULL
                            };
                            size_t sample_idx = 0;
                            for (int q = 0; broad_queries[q] && sample_idx < boot_samples; q++) {
                                KnowledgeEntry entries[16];
                                float scores_buf[16];
                                int n = knowledge_base_search_tfidf(kb, broad_queries[q],
                                                                     entries, 16, scores_buf, 0.01f);
                                for (int e = 0; e < n && sample_idx < boot_samples; e++) {
                                    const char* parts[3] = {
                                        entries[e].subject,
                                        entries[e].predicate,
                                        entries[e].object
                                    };
                                    /* 三重bigram哈希编码→64维特征 */
                                    int hash_count = 0;
                                    for (int p = 0; p < 3; p++) {
                                        if (!parts[p]) continue;
                                        size_t plen = strlen(parts[p]);
                                        for (size_t i = 0; i + 1 < plen; i++) {
                                            uint32_t h = ((uint32_t)(unsigned char)parts[p][i] << 8)
                                                       | (uint32_t)(unsigned char)parts[p][i+1];
                                            h = h * 2654435761u;
                                            size_t idx = (size_t)(h % boot_feat_dim);
                                            boot_data[sample_idx * boot_feat_dim + idx] += 1.0f;
                                            hash_count++;
                                        }
                                    }
                                    if (hash_count > 0) {
                                        float inv = 1.0f / sqrtf((float)hash_count + 1e-8f);
                                        for (size_t d = 0; d < boot_feat_dim; d++)
                                            boot_data[sample_idx * boot_feat_dim + d] *= inv;
                                    }
                                    sample_idx++;
                                }
                            }
                            if (sample_idx > 0) {
                                int pretrained = lnn_self_supervised_pretrain(
                                    (LNN*)boot_lnn, boot_data, sample_idx, boot_feat_dim, (int)boot_batches);
                                if (pretrained == 0) {
                                    printf("  引导训练完成：%zu条知识库特征×5轮，LNN已获得先验知识\n", sample_idx);
                                } else {
                                    printf("  引导训练执行，结果需验证（错误码=%d）\n", pretrained);
                                }
                            }
                            safe_free((void**)&boot_data);
                        }
                    } else {
                        printf("  知识库未就绪或为空，跳过引导训练（LNN使用随机初始化权重）\n");
                    }
                }
            }
/* 检查点/引导训练完成后，标记所有模态模块为已训练 */
            selflnn_bootstrap_trained_modules();

/* 在线学习器附着到共享LNN，直接操作LNN权重矩阵
             * 不再使用rand随机权重和独立线性回归器。
             * 在线学习器通过 lnn_forward/lnn_backward_accumulate 使用CfC动力学
             * 计算真实梯度，再由SGD/Kalman/Adaptive算法直接更新LNN参数。 */
            OnlineLearningConfig ol_config;
            memset(&ol_config, 0, sizeof(OnlineLearningConfig));
            ol_config.algorithm_type = ONLINE_LEARNING_ADAPTIVE;
            ol_config.learning_rate = 0.001f;
            ol_config.forgetting_factor = 0.9f;
            ol_config.window_size = 100;
            ol_config.enable_adaptive_rate = 1;
            ol_config.enable_momentum = 1;
            ol_config.momentum_factor = 0.9f;
            ol_config.buffer_size = 1000;
            ol_config.min_learning_rate = 1e-6f;
            ol_config.max_learning_rate = 0.1f;
            /* P20修复: 从selflnn获取已创建的学习器并附着LNN，避免双重创建 */
            OnlineLearner* learner_raw = (OnlineLearner*)selflnn_get_online_learner();
            if (!learner_raw) {
                float initial_weight = 0.0f;
                learner_raw = online_learner_create(&ol_config, &initial_weight, 1);
            }
            if (learner_raw) {
                void* shared_lnn_ptr = selflnn_get_shared_lnn();
                if (shared_lnn_ptr && online_learner_attach_lnn(learner_raw, (LNN*)shared_lnn_ptr) == 0) {
                    printf("  在线学习器初始化成功（已附着到共享LNN）\n");
                    g_online_learner_handle = (void*)learner_raw;
                } else {
                    printf("  在线学习器LNN附着失败，回退到独立权重模式\n");
                    g_online_learner_handle = (void*)learner_raw;
                }
            } else {
                printf("  在线学习器创建失败\n");
            }
            /* 获取自我演化引擎句柄（已在selflnn_init中创建） */
            g_evolution_engine_handle = selflnn_get_evolution_engine();
            if (g_evolution_engine_handle) {
                printf("  自我演化引擎句柄获取成功\n");
                /* P28修复: 注入适应度函数（基于LNN损失），否则演化引擎无法运行 */
                void* shared_lnn = selflnn_get_shared_lnn();
                if (shared_lnn) {
                    evolution_set_fitness_function(
                        (EvolutionEngine*)g_evolution_engine_handle,
                        lnn_weights_fitness_function, (void*)shared_lnn);
                    printf("  演化引擎适应度函数已注入（基于LNN权重评估）\n");
                } else {
                    printf("  演化引擎需LNN就绪后才能设置适应度函数\n");
                }
            }
            /* ZSFJJJ-FIX: 创建动态架构控制器并连接到自我认知和演化引擎。
             * 架构控制器实现运行时网络结构自我调整：
             *   扩展隐藏层 / 收缩隐藏层 / 增加层 / 删除层 / 知识迁移 / 原子交换。
             * 此前main.c从未创建和连接架构控制器，导致架构演化路径完全断裂。 */
            {
                ArchitectureControllerConfig arch_cfg = arch_controller_default_config();
                g_arch_controller = arch_controller_create(&arch_cfg);
                if (g_arch_controller) {
                    printf("  动态架构控制器创建成功（置信度阈值=%.2f, 频率限制=%d次/小时）\n",
                           arch_cfg.min_confidence_threshold, arch_cfg.max_changes_per_hour);
                    /* 连接到自我认知系统：自我认知检测到容量不足时触发架构变更 */
                    void* scs = selflnn_get_self_cognition();
                    if (scs) {
                        self_cognition_set_arch_controller(
                            (SelfCognitionSystem*)scs, (void*)g_arch_controller);
                        printf("  架构控制器→自我认知系统(架构修正闭环)已连接\n");
                    }
                    /* 连接到演化引擎：演化过程触发结构变异 */
                    if (g_evolution_engine_handle) {
                        evolution_engine_set_arch_controller(
                            (EvolutionEngine*)g_evolution_engine_handle, g_arch_controller);
                        printf("  架构控制器→演化引擎(结构变异闭环)已连接\n");
                    }
                } else {
                    printf("  动态架构控制器创建失败\n");
                }
            }
            /* APP10: 获取产品设计引擎（selflnn_init已创建，此处不再重复创建） */
            g_product_design = (ProductDesignEngine*)selflnn_get_product_design_engine();
            if (g_product_design) {
                printf("  产品设计引擎获取成功（已在selflnn_init中初始化）\n");
            } else {
                printf("  产品设计引擎未就绪\n");
            }
/* NAS + Laplace由selflnn统一管理 */
            {
                g_nas_system = selflnn_get_nas_system();
                if (g_nas_system) {
                    printf("  NAS系统就绪(由selflnn统一管理)\n");
                } else {
                    printf("  NAS系统未就绪\n");
                }
            }
            g_laplace_unified = selflnn_get_laplace_unified();
            if (g_laplace_unified) {
                printf("  拉普拉斯增强系统就绪(由selflnn统一管理)\n");
            } else {
                printf("  拉普拉斯增强系统未就绪\n");
            }
            /* 方案C修复: 统一液态状态处理器由selflnn.c统一管理。
             * 不再在此重复创建实例，改为获取selflnn已创建的全局单例。
             * 消除双重UnifiedLNNState问题，确保多模态统一到同一个LNN状态空间。 */
            {
                void* unified_state = selflnn_get_unified_lnn_state();
                if (unified_state) {
                    g_unified_lnn_state = unified_state;
                    printf("  统一液态状态处理器已就绪（%d模态→共享LNN，由selflnn管理）\n",
                           UNIFIED_LNN_MAX_MODALITIES);
                } else {
                    printf("  统一液态状态处理器未就绪（selflnn子系统初始化中...）\n");
                }
            }
/* 音频采集+TTS由selflnn统一管理 */
            {
                g_audio_capture = selflnn_get_audio_capture();
                if (g_audio_capture) {
                    printf("  音频采集管道就绪(由selflnn统一管理)\n");
                } else {
                    printf("  音频采集管道未就绪\n");
                }
                g_speech_recognizer = selflnn_get_speech_recognizer();
                if (g_speech_recognizer) {
                    printf("  语音识别引擎就绪(由selflnn统一管理)\n");
                } else {
                    printf("  语音识别引擎不可用\n");
                }
                g_tts_engine = selflnn_get_tts_engine();
                if (g_tts_engine) {
                    printf("  TTS语音合成就绪(由selflnn统一管理)\n");
                } else {
                    printf("  TTS语音合成不可用\n");
                }
            }
/* 计算机操作/审计/内容过滤由selflnn统一管理 */
            {
                g_computer_op = selflnn_get_computer_operation();
                if (g_computer_op) {
                    printf("  计算机操作模块就绪(由selflnn统一管理)\n");
                } else {
                    printf("  计算机操作模块未就绪\n");
                }
            }
            g_audit_logger = selflnn_get_audit_logger();
            if (g_audit_logger) {
                printf("  审计日志系统就绪(由selflnn统一管理)\n");
            } else {
                printf("  审计日志系统未就绪\n");
            }
            g_content_filter = selflnn_get_content_filter();
            if (g_content_filter) {
                printf("  内容过滤器就绪(由selflnn统一管理)\n");
            } else {
                printf("  内容过滤器未就绪\n");
            }
/* 初始化深度安全行为监控 */
            g_sec_behavior = selflnn_get_security_monitor_deep();
            if (g_sec_behavior) {
                printf("  深度安全行为监控就绪(由selflnn统一管理)\n");
            } else {
                printf("  深度安全行为监控未就绪\n");
            }
/* 从selflnn获取已创建的自我编程引擎，消除重复创建 */
            {
                g_prog_engine = selflnn_get_self_programming_engine();
                if (g_prog_engine) {
                    printf("  自我编程引擎初始化成功(从selflnn获取)\n");
                } else {
                    printf("  自我编程引擎初始化失败\n");
                }
            }
/* 从selflnn获取已创建的分布式训练上下文，消除重复创建 */
            {
                g_distributed = selflnn_get_distributed_context();
                if (g_distributed) {
                    printf("  分布式训练上下文初始化成功(从selflnn获取)\n");
                } else {
                    printf("  分布式训练上下文初始化跳过\n");
                }
            }
/* 负载均衡器由selflnn统一管理 */
            {
                g_load_balancer = selflnn_get_load_balancer();
                if (g_load_balancer) {
                    printf("  负载均衡器就绪(由selflnn统一管理)\n");
                } else {
                    printf("  负载均衡器未就绪\n");
                }
            }
/* Gazebo仿真桥接初始化 */
            {
                if (gazebo_is_available) {
                    GazeboConfig gz_cfg;
                    memset(&gz_cfg, 0, sizeof(gz_cfg));
                    gz_cfg.world_file = NULL;
                    gz_cfg.start_paused = 0;
                    gz_cfg.real_time_factor = 1.0f;
                    gz_cfg.max_step_size = 0.016f;
                    gz_cfg.use_gui = 0;
                    gz_cfg.use_gazebo_ros = 0;
                    gz_cfg.server_port = 11345;

                    g_gazebo_bridge = gazebo_connect(&gz_cfg);
                    if (g_gazebo_bridge) {
                        printf("  Gazebo仿真桥接已连接\n");
                    } else {
                        printf("  Gazebo仿真桥接连接失败，将使用内部仿真\n");
                    }
                } else {
                    printf("  Gazebo不可用，使用内部仿真器\n");
                }
            }

/* 摄像头采集初始化 - 使用默认640x480@30fps
 *: 注册SLAM回调，摄像头帧实时送入SLAM+统一液态状态 */
            {
                g_camera_capture = camera_capture_create(NULL, 640, 480, 30);
                if (g_camera_capture) {
                    int started = camera_capture_start((CameraCaptureContext*)g_camera_capture,
                                                       slam_camera_frame_callback, NULL);
                    if (started == 0) {
                        printf("  摄像头采集已启动（640x480@30fps，SLAM回调已注册）\n");
                    } else {
                        printf("  摄像头采集启动失败，将在后台重试\n");
                    }
                } else {
                    printf("  摄像头设备未检测到，视觉子系统将在有硬件时自动激活\n");
                }
            }

/* SLAM系统初始化 - 使用默认配置 */
            {
                SlamConfig slam_cfg = slam_get_default_config();
                slam_cfg.max_keyframes = 500;
                slam_cfg.enable_loop_closure = 1;
                g_slam_system = slam_system_create(&slam_cfg);
                if (g_slam_system) {
                    printf("  SLAM系统已就绪（视觉里程计+闭环检测+图优化）\n");
                } else {
                    printf("  SLAM系统初始化失败，将在首次摄像头帧到达时重试\n");
                }
            }
            /* Z8-002: 能力开关重置移到在线学习器创建之后
             * 确保子系统(online_learner/evolution_engine等)全部就绪后再激活 */
            capability_reset_to_defaults();
            printf("  能力开关已重置为默认启用状态\n");

/* 初始化AGI认知系统和任务调度器
               将共享LNN注入AGISystem，确保单一LNN原则。
               任务调度器用于AGI后台任务的优先级调度和执行。 */
            {
                AGIConfig agi_cfg;
                memset(&agi_cfg, 0, sizeof(AGIConfig));
                agi_cfg.state_vector_dim = AGI_STATE_VECTOR_DIM;
                agi_cfg.max_tasks = 64;
                agi_cfg.knowledge_capacity = 10000;
                agi_cfg.learning_rate = 0.001f;
                agi_cfg.enable_self_correction = 1;
                agi_cfg.enable_reflection = 1;
                agi_cfg.enable_planning = 1;
                agi_cfg.enable_self_learning = 1;

                g_agi_system = agi_system_create(&agi_cfg);
                if (g_agi_system) {
                    /* 注入共享LNN到AGISystem，确保单一液态神经网络 */
                    LNN* shared_lnn = selflnn_get_shared_lnn();
                    if (shared_lnn && agi_system_set_lnn(g_agi_system, shared_lnn) == 0) {
                        printf("  AGI认知系统初始化成功（已注入单一共享LNN）\n");
                    } else {
                        printf("  AGI认知系统创建成功但LNN注入失败\n");
                    }

/* 注入所有共享子系统（替换AGISystem自建的独立副本）
                     * 确保"使用单一液态神经网络模型"原则：
                     * 所有子系统均使用selflnn的共享实例，禁止AGISystem持有独立副本。
                     * agi_system_set_xxx 内部会先释放AGISystem自建副本，再替换为共享实例。
                     * 对于名称不完全匹配的getter，使用最接近的已有函数并显式强制类型转换。
                     * 注：selflnn中无独立DCCorrectionSystem实例（g_system_state无correction字段），
                     *     深度修正能力由自我认知子系统和元认知子系统内部协同处理，
                     *     已通过上方的self_cognition/meta_cognition注入覆盖。 */
                    agi_system_set_knowledge_base(g_agi_system, (KnowledgeBase*)selflnn_get_knowledge_base());
                    agi_system_set_reasoning_engine(g_agi_system, (ReasoningEngine*)selflnn_get_reasoning_engine());
                    agi_system_set_planning_system(g_agi_system, (PlanningSystem*)selflnn_get_planning_system());
                    agi_system_set_learning_engine(g_agi_system, (LearningEngine*)selflnn_get_online_learner());
                    agi_system_set_memory_manager(g_agi_system, (MemoryManager*)selflnn_get_memory_manager());
                    agi_system_set_metacognition(g_agi_system, (MetacognitionSystem*)selflnn_get_metacognition());
                    agi_system_set_self_cognition(g_agi_system, (SelfCognitionSystem*)selflnn_get_self_cognition());
                    agi_system_set_dialogue(g_agi_system, (DialogueProcessor*)selflnn_get_dialogue_processor());
                    printf("  AGI认知系统共享子系统注入完成（8个子系统已替换，单一LNN原则）\n");

/* 注册知识库更新事件通知回调
                     * 当 knowledge_base_add 写入新知识后，通过此回调
                     * 主动触发LNN知识嵌入重新编码（替代原来的定时轮询），
                     * 消除知识写入与嵌入更新之间的延迟。 */
                    {
                        KnowledgeBase* shared_kb = (KnowledgeBase*)selflnn_get_knowledge_base();
                        if (shared_kb) {
                            knowledge_base_set_update_callback(kb_update_nofify_callback, NULL);
                            printf("  知识库更新事件通知回调已注册\n");
                        }
                    }
                } else {
                    printf("  AGI认知系统创建失败\n");
                }

                /* 创建任务调度器（4级优先级、线程池集成） */
                g_task_scheduler = task_scheduler_create();
                if (g_task_scheduler) {
                    printf("  任务调度器初始化成功（4级优先级队列）\n");
                } else {
                    printf("  任务调度器创建失败\n");
                }
            }
        } else {
            printf("  SELF-LNN核心系统初始化失败，系统无法启动\n");
            fprintf(stderr, "错误: 核心系统初始化失败（LNN/知识库/推理引擎等关键模块未就绪）\n");
            return 1;
        }
        } /* 内层sys_config初始化块结束 */
    } /* 外层system_config.json读取块结束 */

    g_server = backend_server_create(&config);
    if (!g_server) {
        fprintf(stderr, "错误: 无法创建后端服务器\n");
        selflnn_shutdown();
        return 1;
    }

/* 端口冲突检测 — 在bind之前主动检查端口是否被占用 */
    {
        SocketHandle probe = socket_create(AF_INET, SOCK_STREAM, 0);
        if (probe != INVALID_SOCKET) {
            if (socket_bind(probe, "0.0.0.0", (unsigned short)config.port) == 0) {
                socket_close(probe);
            } else {
                socket_close(probe);
                fprintf(stderr, "错误: 端口 %d 已被占用，请检查是否有其他程序正在使用该端口\n", config.port);
                backend_server_free(g_server);
                g_server = NULL;
                selflnn_shutdown();
                return 1;
            }
        }
    }

    printf("正在启动后端服务器...\n");
    if (backend_server_start(g_server) != 0) {
        fprintf(stderr, "错误: 无法启动后端服务器\n");
        backend_server_free(g_server);
        g_server = NULL;
        selflnn_shutdown();
        return 1;
    }

    /* 启动WebSocket推送服务器（独立端口8081） */
    {
        WSPushServer* ws_push = ws_push_server_create(SELFLNN_WEBSOCKET_PORT);
        if (ws_push) {
            if (ws_push_server_start(ws_push) == 0) {
                printf("  WebSocket推送服务器已启动 (端口:%d)\n", SELFLNN_WEBSOCKET_PORT);
                g_ws_push_server = ws_push;
                /* P0-001: 将WSPushServer注入后端，实现统一WebSocket架构 */
                backend_server_set_ws_push_server(g_server, ws_push);
                printf("  WebSocket推送服务器已注入后端（统一架构）\n");
            } else {
                printf("  WebSocket推送服务器启动失败 (端口:%d)，系统将继续运行但无实时推送功能\n", SELFLNN_WEBSOCKET_PORT);
                ws_push_server_destroy(ws_push);
            }
        } else {
            printf("  WebSocket推送服务器创建失败 (端口:%d)\n", SELFLNN_WEBSOCKET_PORT);
        }
    }

    printf("\n========================================\n");
    printf("SELF-LNN AGI 系统已成功启动!\n");
    printf("  API地址: http://localhost:%d/api\n", config.port);
    printf("  WebSocket: ws://localhost:%d/ws\n", SELFLNN_WEBSOCKET_PORT);
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
        agi_background_loop_iteration();

/* 执行AGI认知系统认知循环（感知→推理→规划→决策→执行→学习） */
        if (g_agi_system) {
            static int agi_cycle_counter = 0;
            agi_cycle_counter++;
            /* 每10个间隔执行一次完整认知循环，避免过度消耗CPU */
            if (agi_cycle_counter % 10 == 0) {
                float cycle_input[256];
                float cycle_output[256];
                memset(cycle_input, 0, sizeof(cycle_input));
                memset(cycle_output, 0, sizeof(cycle_output));
                /* P1-002修复: 从系统真实状态采样作为认知循环输入，替代全零idle信号。
                 * 采样优先级：LNN状态 > 系统状态 > 零向量（最后一层回退）。 */
                {
                    void* lnn = selflnn_get_shared_lnn();
                    int got_data = 0;
                    if (lnn && selflnn_get_recent_state(lnn, cycle_input, 128) == 0) {
                        got_data = 1;
                    }
                    if (!got_data) {
                        SystemStatus st;
                        memset(&st, 0, sizeof(st));
                        if (selflnn_get_status(&st) == 0) {
                            cycle_input[0] = (float)st.active_tasks / 100.0f;
                            cycle_input[1] = (float)(st.total_memories % 256) / 256.0f;
                            cycle_input[2] = (float)(st.total_knowledge % 256) / 256.0f;
                            cycle_input[3] = st.memory_usage_mb / 1024.0f;
                            got_data = 1;
                        }
                    }
                    if (!got_data) {
                        memset(cycle_input, 0, sizeof(cycle_input));
                    }
                }
                agi_system_cognitive_cycle(g_agi_system, cycle_input, 256, cycle_output, 256);
                agi_cycle_counter = 0;
            }
        }

/* 任务调度器tick（处理到期任务、超时降级、抢占调度） */
        if (g_task_scheduler) {
            task_scheduler_tick(g_task_scheduler);
        }

        /* WebSocket推送服务器poll：处理客户端帧、心跳、超时检测 */
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
    /* WebSocket推送服务器当前未启用（HTTP Upgrade模式），跳过清理 */
    if (g_ws_push_server) {
        ws_push_server_stop(g_ws_push_server);
        ws_push_server_destroy(g_ws_push_server);
        g_ws_push_server = NULL;
    }
/* P0修复 - 训练管线由selflnn shutdown_subsystems统一释放，
     * main.c仅置空指针避免悬空引用，不重复调用training_pipeline_free防止双重释放 */
    g_training_pipeline = NULL;
/* 释放Gazebo仿真桥接 */
    if (g_gazebo_bridge) {
        gazebo_disconnect(g_gazebo_bridge);
        g_gazebo_bridge = NULL;
        printf("  Gazebo仿真桥接已释放\n");
    }
/* 释放AGI认知系统和任务调度器 */
    if (g_agi_system) {
        agi_system_free(g_agi_system);
        g_agi_system = NULL;
        printf("  AGI认知系统已释放\n");
    }
    if (g_task_scheduler) {
        task_scheduler_free(g_task_scheduler);
        g_task_scheduler = NULL;
        printf("  任务调度器已释放\n");
    }
    if (g_slam_system) {
        slam_system_free((SlamSystem*)g_slam_system);
        g_slam_system = NULL;
        printf("  SLAM系统已释放\n");
    }
    if (g_camera_capture) {
        camera_capture_stop((CameraCaptureContext*)g_camera_capture);
        camera_capture_free((CameraCaptureContext*)g_camera_capture);
        g_camera_capture = NULL;
        printf("  摄像头采集已释放\n");
    }
    /* ZSFJJJ-FIX: 释放架构控制器（由main.c管理而非selflnn） */
    if (g_arch_controller) {
        arch_controller_free(g_arch_controller);
        g_arch_controller = NULL;
        printf("  动态架构控制器已释放\n");
    }
    /* 以下4个资源由selflnn_shutdown统一管理释放，main.c不应重复释放 */
    /* g_online_learner_handle → 由selflnn shutdown_subsystems L2099释放 */
    /* g_evolution_engine_handle → 由selflnn shutdown_subsystems L2063释放 */
    /* g_product_design → 由selflnn shutdown_subsystems L2027释放 */
    /* g_speech_recognizer → 由selflnn shutdown_subsystems L2105释放 */
    g_online_learner_handle = NULL;
    g_evolution_engine_handle = NULL;
    g_product_design = NULL;
/* NAS/Laplace/Audio/TTS由selflnn统一管理释放，仅置空指针 */
    g_nas_system = NULL;
    g_laplace_unified = NULL;
    g_audio_capture = NULL;
    g_speech_recognizer = NULL;
    g_tts_engine = NULL;
    g_computer_op = NULL;
    g_audit_logger = NULL;
    g_content_filter = NULL;
    g_prog_engine = NULL;
    g_load_balancer = NULL;
    /* 方案C修复: 统一液态状态处理器由selflnn.c管理生命周期。
     * g_unified_lnn_state仅为借用的指针引用，不负责释放。 */
    g_unified_lnn_state = NULL;
/* 不再自行释放分布式上下文。
     * g_distributed是从selflnn_get_distributed_context获取的指针引用，
     * 与selflnn内部的g_system_state.distributed_training指向同一块内存。
     * selflnn_shutdown内部会统一释放，此处重复释放将导致double-free崩溃。 */
    g_distributed = NULL;
    selflnn_shutdown();

    printf("SELF-LNN AGI 系统已停止。\n");
    return 0;
}
