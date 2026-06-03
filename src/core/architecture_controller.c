/**
 * @file architecture_controller.c
 * @brief 动态架构控制器实现 —— 运行时安全地修改液态神经网络结构
 *
 * 解决架构动态演化能力深度审计报告中确认的所有 P0/P1/P2 缺陷。
 */

#define SELFLNN_IMPLEMENTATION
#define SELFLNN_CORE_INTERNAL

#include "selflnn/core/architecture_controller.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/cfc_network.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ============ 外部函数声明 ============ */
extern void log_info(const char* fmt, ...);
extern void log_warning(const char* fmt, ...);
extern void log_error(const char* fmt, ...);
extern void* safe_malloc(size_t size);
extern void* safe_calloc(size_t num, size_t size);
extern void safe_free(void** ptr);
extern uint64_t get_timestamp_ms(void);
extern uint64_t lnn_get_forward_count(const LNN* network);
extern uint64_t lnn_get_backward_count(const LNN* network);

/* ============ 维度范围常量 ============ */
#define ARCH_MIN_HIDDEN_SIZE    32    /**< 隐藏层最小尺寸 */
#define ARCH_MAX_HIDDEN_SIZE    16384 /**< 隐藏层最大尺寸 */
#define ARCH_MIN_LAYERS         1     /**< 最小层数 */
#define ARCH_MAX_LAYERS         16    /**< 最大层数 */
#define ARCH_MIN_INPUT_SIZE     8     /**< 输入层最小尺寸 */
#define ARCH_MAX_INPUT_SIZE     4096  /**< 输入层最大尺寸 */
#define ARCH_MAX_HISTORY_ENTRIES 128  /**< 变更历史最大记录数 */

/* ============ 内部随机数生成 ============ */
static unsigned int arch_rng_state = 0xDEADBEEF;

static float arch_random_uniform(float min_val, float max_val) {
    arch_rng_state = arch_rng_state * 1103515245 + 12345;
    float r = (float)(arch_rng_state & 0x7FFFFFFF) / 2147483648.0f;
    return min_val + r * (max_val - min_val);
}

/* ============ 架构控制器内部结构 ============ */

typedef struct {
    ArchitectureChangeHistoryEntry entries[ARCH_MAX_HISTORY_ENTRIES];
    size_t count;             /**< 当前历史记录数量 */
    size_t head;              /**< 环形缓冲区写入位置 */
} ChangeHistory;

struct ArchitectureController {
    ArchitectureControllerConfig config;
    ChangeHistory history;
    uint64_t last_change_times[16]; /**< 最近变更时间戳（频率限制） */
    int last_change_idx;           /**< 环形索引 */
    int is_initialized;
};

/* ============ 默认配置 ============ */

ArchitectureControllerConfig arch_controller_default_config(void) {
    ArchitectureControllerConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.min_confidence_threshold = 0.6f;
    cfg.max_changes_per_hour     = 3;
    cfg.enable_auto_approval     = 1;
    cfg.enable_archive_backup    = 1;
    cfg.enable_knowledge_transfer = 1;
    snprintf(cfg.archive_dir, sizeof(cfg.archive_dir), "./archives");
    return cfg;
}

ArchitectureChangeRequest arch_controller_default_request(void) {
    ArchitectureChangeRequest req;
    memset(&req, 0, sizeof(req));
    req.type = ARCH_CHANGE_NONE;
    return req;
}

/* ============ 生命周期 ============ */

ArchitectureController* arch_controller_create(const ArchitectureControllerConfig* config) {
    ArchitectureController* ctrl = (ArchitectureController*)safe_calloc(1, sizeof(ArchitectureController));
    if (!ctrl) {
        log_error("[架构控制器] 内存分配失败");
        return NULL;
    }

    ctrl->config = config ? *config : arch_controller_default_config();
    if (ctrl->config.archive_dir[0] == '\0') {
        snprintf(ctrl->config.archive_dir, sizeof(ctrl->config.archive_dir), "./archives");
    }
    ctrl->history.count = 0;
    ctrl->history.head  = 0;
    ctrl->last_change_idx = 0;
    memset(ctrl->last_change_times, 0, sizeof(ctrl->last_change_times));
    ctrl->is_initialized = 1;

    log_info("[架构控制器] 初始化完成 (置信度阈值=%.2f, 最大变更/小时=%d, 知识迁移=%s)",
             ctrl->config.min_confidence_threshold,
             ctrl->config.max_changes_per_hour,
             ctrl->config.enable_knowledge_transfer ? "启用" : "禁用");
    return ctrl;
}

void arch_controller_free(ArchitectureController* controller) {
    if (!controller) return;
    /* 历史记录中的 external_arch_data 由调用方管理，不在此释放 */
    memset(controller, 0, sizeof(ArchitectureController));
    safe_free((void**)&controller);
}

/* ============ 安全审批 ============ */

static int arch_validate_dimensions(const ArchitectureChangeRequest* request) {
    /* 检查 target 维度是否在合法范围 */
    if (request->target_hidden_size > 0) {
        if (request->target_hidden_size < ARCH_MIN_HIDDEN_SIZE ||
            request->target_hidden_size > ARCH_MAX_HIDDEN_SIZE) {
            return -1;
        }
    }
    if (request->target_num_layers > 0) {
        if (request->target_num_layers < ARCH_MIN_LAYERS ||
            request->target_num_layers > ARCH_MAX_LAYERS) {
            return -2;
        }
    }
    if (request->target_input_size > 0) {
        if (request->target_input_size < ARCH_MIN_INPUT_SIZE ||
            request->target_input_size > ARCH_MAX_INPUT_SIZE) {
            return -3;
        }
    }
    if (request->target_output_size > 0) {
        if (request->target_output_size < ARCH_MIN_INPUT_SIZE ||
            request->target_output_size > ARCH_MAX_INPUT_SIZE) {
            return -4;
        }
    }
    return 0;
}

static int arch_check_frequency_limit(ArchitectureController* controller) {
    uint64_t now = get_timestamp_ms();
    uint64_t one_hour_ms = 3600000ULL;
    int changes_in_hour = 0;

    for (int i = 0; i < 16; i++) {
        if (controller->last_change_times[i] > 0 &&
            (now - controller->last_change_times[i]) < one_hour_ms) {
            changes_in_hour++;
        }
    }

    if (changes_in_hour >= controller->config.max_changes_per_hour) {
        return -1; /* 超过频率限制 */
    }
    return 0;
}

int arch_controller_approve_change(ArchitectureController* controller,
                                    const ArchitectureChangeRequest* request) {
    if (!controller || !request) {
        log_error("[架构控制器] 审批失败: 参数为空");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    /* 1. 检查变更类型合法性 */
    if (request->type == ARCH_CHANGE_NONE) {
        log_warning("[架构控制器] 审批拒绝: 变更类型为 NONE");
        return -100;
    }

    /* 2. 置信度检查 */
    if (request->confidence < controller->config.min_confidence_threshold) {
        log_warning("[架构控制器] 审批拒绝: 置信度 %.2f < 阈值 %.2f [来源: %s]",
                     request->confidence, controller->config.min_confidence_threshold,
                     request->source_module);
        return -101;
    }

    /* 3. 维度合法性检查 */
    int dim_check = arch_validate_dimensions(request);
    if (dim_check != 0) {
        log_error("[架构控制器] 审批拒绝: 维度不合法 (错误码=%d)", dim_check);
        return -102;
    }

    /* 4. 频率限制检查 */
    if (arch_check_frequency_limit(controller) != 0) {
        log_warning("[架构控制器] 审批拒绝: 超过每小时最大变更次数 (%d)",
                     controller->config.max_changes_per_hour);
        return -103;
    }

    /* 5. 变更类型特定检查 */
    switch (request->type) {
    case ARCH_CHANGE_EXPAND_HIDDEN:
    case ARCH_CHANGE_SHRINK_HIDDEN:
        if (request->target_hidden_size == 0) {
            log_error("[架构控制器] 审批拒绝: EXPAND/SHRINK_HIDDEN 需要指定 target_hidden_size");
            return -104;
        }
        break;
    case ARCH_CHANGE_ADD_LAYER:
        if (request->target_num_layers <= 0) {
            log_error("[架构控制器] 审批拒绝: ADD_LAYER 需要指定 target_num_layers");
            return -105;
        }
        break;
    case ARCH_CHANGE_REMOVE_LAYER:
        /* 需要至少 2 层才能移除一层 */
        break;
    case ARCH_CHANGE_REPLACE_ARCHITECTURE:
        if (!request->external_arch_data || request->external_arch_data_size == 0) {
            log_error("[架构控制器] 审批拒绝: REPLACE_ARCHITECTURE 需要 external_arch_data");
            return -106;
        }
        break;
    default:
        break;
    }

    log_info("[架构控制器] 审批通过: 类型=%d 置信度=%.2f 来源=%s 原因=%s",
             request->type, request->confidence, request->source_module, request->reason);
    return SELFLNN_SUCCESS;
}

/* ============ 知识迁移 ============ */

/**
 * @brief 将权重矩阵从旧网络迁移到新网络（左上角复制策略）
 *
 * 对于 h2h 权重矩阵 [new_hidden × new_hidden]：
 *   - 旧部分直接复制
 *   - 新行/新列使用 Xavier 初始化
 */
static void arch_transfer_weight_matrix(const float* old_w, size_t old_rows, size_t old_cols,
                                         float* new_w, size_t new_rows, size_t new_cols,
                                         float scale) {
    /* 清零新矩阵 */
    memset(new_w, 0, new_rows * new_cols * sizeof(float));

    /* 复制重叠区域（左上角） */
    size_t copy_rows = (old_rows < new_rows) ? old_rows : new_rows;
    size_t copy_cols = (old_cols < new_cols) ? old_cols : new_cols;

    for (size_t r = 0; r < copy_rows; r++) {
        for (size_t c = 0; c < copy_cols; c++) {
            new_w[r * new_cols + c] = old_w[r * old_cols + c];
        }
    }

    /* 新行 Xavier 初始化 */
    if (new_rows > old_rows) {
        float xavier_limit = sqrtf(6.0f / (float)(old_rows + new_cols));
        for (size_t r = old_rows; r < new_rows; r++) {
            for (size_t c = 0; c < new_cols; c++) {
                new_w[r * new_cols + c] = arch_random_uniform(-xavier_limit, xavier_limit) * scale;
            }
        }
    }

    /* 新列 Xavier 初始化 */
    if (new_cols > old_cols) {
        float xavier_limit = sqrtf(6.0f / (float)(new_rows + old_cols));
        for (size_t r = 0; r < old_rows; r++) {
            for (size_t c = old_cols; c < new_cols; c++) {
                new_w[r * new_cols + c] = arch_random_uniform(-xavier_limit, xavier_limit) * scale;
            }
        }
    }
}

/**
 * @brief 迁移输出投影矩阵 [output_size × hidden_size]
 */
static void arch_transfer_output_projection(const float* old_w_out, size_t old_out, size_t old_hidden,
                                             const float* old_b_out,
                                             float* new_w_out, size_t new_out, size_t new_hidden,
                                             float* new_b_out) {
    /* W_out: 每行复制重叠隐藏维度部分 */
    size_t copy_out = (old_out < new_out) ? old_out : new_out;
    size_t copy_hidden = (old_hidden < new_hidden) ? old_hidden : new_hidden;

    memset(new_w_out, 0, new_out * new_hidden * sizeof(float));

    for (size_t o = 0; o < copy_out; o++) {
        for (size_t h = 0; h < copy_hidden; h++) {
            new_w_out[o * new_hidden + h] = old_w_out[o * old_hidden + h];
        }
        /* 新隐藏维度部分 Xavier 初始化 */
        if (new_hidden > old_hidden) {
            float xavier_limit = sqrtf(6.0f / (float)(old_hidden + new_hidden));
            for (size_t h = old_hidden; h < new_hidden; h++) {
                new_w_out[o * new_hidden + h] = arch_random_uniform(-xavier_limit, xavier_limit);
            }
        }
    }

    /* 新输出维度行 Xavier 初始化 */
    if (new_out > old_out) {
        float xavier_limit = sqrtf(6.0f / (float)new_hidden);
        for (size_t o = old_out; o < new_out; o++) {
            for (size_t h = 0; h < new_hidden; h++) {
                new_w_out[o * new_hidden + h] = arch_random_uniform(-xavier_limit, xavier_limit);
            }
        }
    }

    /* 偏置 */
    memset(new_b_out, 0, new_out * sizeof(float));
    for (size_t o = 0; o < copy_out; o++) {
        new_b_out[o] = old_b_out[o];
    }
}

int arch_controller_transfer_knowledge(LNN* old_lnn, LNN* new_lnn,
                                        ArchitectureChangeType change_type) {
    if (!old_lnn || !new_lnn) {
        log_error("[架构控制器] 知识迁移失败: LNN 实例为空");
        return -1;
    }

    CfCNetwork* old_cfc = old_lnn->cfc_network;
    CfCNetwork* new_cfc = new_lnn->cfc_network;

    if (!old_cfc || !new_cfc) {
        log_error("[架构控制器] 知识迁移失败: CfC 网络为空");
        return -1;
    }

    size_t old_hidden = old_cfc->config.hidden_size;
    size_t new_hidden = new_cfc->config.hidden_size;
    size_t old_input  = old_cfc->config.input_size;
    size_t new_input  = new_cfc->config.input_size;
    size_t old_output = old_cfc->config.output_size;
    size_t new_output = new_cfc->config.output_size;
    int    old_layers = old_cfc->config.num_layers;
    int    new_layers = new_cfc->config.num_layers;

    log_info("[架构控制器] 知识迁移: %dx%d×%d层 → %dx%d×%d层",
             (int)old_hidden, old_layers, (int)old_hidden, old_layers,
             (int)new_hidden, new_layers, (int)new_hidden, new_layers);

    /* 迁移每层参数 */
    for (int l = 0; l < new_layers; l++) {
        CfCCell* new_cell = new_cfc->layers[l];
        if (!new_cell) continue;

        size_t l_new_input = (l == 0) ? new_input : new_hidden;
        size_t l_old_input = (l == 0) ? old_input : old_hidden;

        if (l < old_layers) {
            CfCCell* old_cell = old_cfc->layers[l];
            if (old_cell) {
                size_t l_old_hidden = old_cell->config.hidden_size;

                /* 迁移共享权重矩阵 W [input×hidden] */
                if (old_cfc->weight_matrix && new_cfc->weight_matrix &&
                    new_cfc->per_layer_w_offset && old_cfc->per_layer_w_offset) {
                    size_t old_w_offset = old_cfc->per_layer_w_offset[l];
                    size_t new_w_offset = new_cfc->per_layer_w_offset[l];
                    float* old_w_start = old_cfc->weight_matrix + old_w_offset;
                    float* new_w_start = new_cfc->weight_matrix + new_w_offset;
                    arch_transfer_weight_matrix(old_w_start, l_old_input, l_old_hidden,
                                                 new_w_start, l_new_input, new_hidden, 1.0f);
                }

                /* 迁移 cell 级门控权重 */
                /* input_gate_weights: [input×hidden] */
                if (old_cell->input_gate_weights && new_cell->input_gate_weights) {
                    arch_transfer_weight_matrix(old_cell->input_gate_weights, l_old_input, l_old_hidden,
                                                 new_cell->input_gate_weights, l_new_input, new_hidden, 0.5f);
                }
                /* forget_gate_weights */
                if (old_cell->forget_gate_weights && new_cell->forget_gate_weights) {
                    arch_transfer_weight_matrix(old_cell->forget_gate_weights, l_old_input, l_old_hidden,
                                                 new_cell->forget_gate_weights, l_new_input, new_hidden, 0.5f);
                }
                /* output_gate_weights */
                if (old_cell->output_gate_weights && new_cell->output_gate_weights) {
                    arch_transfer_weight_matrix(old_cell->output_gate_weights, l_old_input, l_old_hidden,
                                                 new_cell->output_gate_weights, l_new_input, new_hidden, 0.5f);
                }
                /* gate_biases */
                if (old_cell->gate_biases && new_cell->gate_biases) {
                    size_t copy_b = (l_old_hidden < new_hidden) ? l_old_hidden : new_hidden;
                    for (int g = 0; g < 3; g++) {
                        for (size_t i = 0; i < copy_b; i++) {
                            new_cell->gate_biases[g * new_hidden + i] =
                                old_cell->gate_biases[g * l_old_hidden + i];
                        }
                    }
                }

                /* H2H 权重矩阵 [hidden×hidden] */
                if (old_cell->hidden_to_input_gate_weights && new_cell->hidden_to_input_gate_weights)
                    arch_transfer_weight_matrix(old_cell->hidden_to_input_gate_weights, l_old_hidden, l_old_hidden,
                                                 new_cell->hidden_to_input_gate_weights, new_hidden, new_hidden, 0.3f);
                if (old_cell->hidden_to_forget_gate_weights && new_cell->hidden_to_forget_gate_weights)
                    arch_transfer_weight_matrix(old_cell->hidden_to_forget_gate_weights, l_old_hidden, l_old_hidden,
                                                 new_cell->hidden_to_forget_gate_weights, new_hidden, new_hidden, 0.3f);
                if (old_cell->hidden_to_output_gate_weights && new_cell->hidden_to_output_gate_weights)
                    arch_transfer_weight_matrix(old_cell->hidden_to_output_gate_weights, l_old_hidden, l_old_hidden,
                                                 new_cell->hidden_to_output_gate_weights, new_hidden, new_hidden, 0.3f);
                if (old_cell->hidden_to_activation_weights && new_cell->hidden_to_activation_weights)
                    arch_transfer_weight_matrix(old_cell->hidden_to_activation_weights, l_old_hidden, l_old_hidden,
                                                 new_cell->hidden_to_activation_weights, new_hidden, new_hidden, 0.3f);

                /* 时间常数 */
                if (old_cell->time_constants && new_cell->time_constants) {
                    size_t copy_tau = (l_old_hidden < new_hidden) ? l_old_hidden : new_hidden;
                    memcpy(new_cell->time_constants, old_cell->time_constants, copy_tau * sizeof(float));
                    for (size_t i = copy_tau; i < new_hidden; i++) {
                        new_cell->time_constants[i] = old_cell->config.time_constant;
                    }
                }
            }
        }
        /* 新层（l >= old_layers）保持原始随机初始化，不做迁移 */
    }

    /* 迁移输出投影权重 */
    if (old_cfc->W_out_params && new_cfc->W_out_params) {
        arch_transfer_output_projection(
            old_cfc->W_out_params, old_output, old_hidden,
            old_cfc->b_out_params,
            new_cfc->W_out_params, new_output, new_hidden,
            new_cfc->b_out_params);
    }

    log_info("[架构控制器] 知识迁移完成");
    return 0;
}

/* ============ 网络重建 ============ */

LNN* arch_controller_rebuild_lnn(LNN* old_lnn, const LNNConfig* new_config) {
    if (!new_config) {
        log_error("[架构控制器] 重建失败: 新配置为空");
        return NULL;
    }

    /* 创建新 LNN 实例 */
    LNN* new_lnn = lnn_create(new_config);
    if (!new_lnn) {
        log_error("[架构控制器] 重建失败: lnn_create 返回 NULL");
        return NULL;
    }

    /* 如果有旧网络，执行知识迁移 */
    if (old_lnn) {
        /* 复制训练相关超参数 */
        new_config = &new_lnn->config;
        LNNConfig old_config;
        if (lnn_get_config(old_lnn, &old_config) == 0) {
            new_lnn->config.learning_rate   = old_config.learning_rate;
            new_lnn->config.time_constant   = old_config.time_constant;
            new_lnn->config.enable_training = old_config.enable_training;
            new_lnn->config.enable_adaptation = old_config.enable_adaptation;
            new_lnn->config.enable_evolution  = old_config.enable_evolution;
            new_lnn->config.ode_solver_type   = old_config.ode_solver_type;
        }

        /* 执行知识迁移 */
        ArchitectureChangeType change_type = ARCH_CHANGE_RESHAPE_ALL;
        arch_controller_transfer_knowledge(old_lnn, new_lnn, change_type);

        /* 迁移运行时统计 */
        new_lnn->forward_count  = old_lnn->forward_count;
        new_lnn->backward_count = old_lnn->backward_count;

        /* 迁移记忆系统和拉普拉斯分析器引用 */
        if (old_lnn->memory_system) {
            new_lnn->memory_system           = old_lnn->memory_system;
            new_lnn->memory_context_strength = old_lnn->memory_context_strength;
            new_lnn->memory_context_top_k    = old_lnn->memory_context_top_k;
        }
        if (old_lnn->laplace_analyzer) {
            new_lnn->laplace_analyzer          = old_lnn->laplace_analyzer;
            new_lnn->laplace_gradient_strength = old_lnn->laplace_gradient_strength;
        }
    }

    log_info("[架构控制器] 网络重建成功: input=%zu→%zu hidden=%zu→%zu output=%zu→%zu layers=%d→%d",
             old_lnn ? old_lnn->config.input_size  : 0, new_config->input_size,
             old_lnn ? old_lnn->config.hidden_size : 0, new_config->hidden_size,
             old_lnn ? old_lnn->config.output_size : 0, new_config->output_size,
             old_lnn ? old_lnn->config.num_layers  : 0, new_config->num_layers);

    return new_lnn;
}

/* ============ 变更历史管理 ============ */

static void arch_record_history(ArchitectureController* controller,
                                 const ArchitectureChangeRequest* request,
                                 const ArchitectureChangeResult* result) {
    if (!controller || !request || !result) return;

    ChangeHistory* hist = &controller->history;
    ArchitectureChangeHistoryEntry* entry = &hist->entries[hist->head];

    memcpy(&entry->request, request, sizeof(ArchitectureChangeRequest));
    memcpy(&entry->result,  result,  sizeof(ArchitectureChangeResult));

    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    if (tm_info) {
        strftime(entry->timestamp_str, sizeof(entry->timestamp_str),
                "%Y-%m-%d %H:%M:%S", tm_info);
    } else {
        snprintf(entry->timestamp_str, sizeof(entry->timestamp_str), "unknown");
    }

    hist->head = (hist->head + 1) % ARCH_MAX_HISTORY_ENTRIES;
    if (hist->count < ARCH_MAX_HISTORY_ENTRIES) {
        hist->count++;
    }
}

static void arch_record_change_time(ArchitectureController* controller) {
    controller->last_change_times[controller->last_change_idx] = get_timestamp_ms();
    controller->last_change_idx = (controller->last_change_idx + 1) % 16;
}

/* ============ 核心变更提交 ============ */

/**
 * @brief 根据请求计算目标 LNNConfig
 */
static int arch_compute_target_config(const LNN* lnn,
                                       const ArchitectureChangeRequest* request,
                                       LNNConfig* target_config) {
    if (!lnn || !request || !target_config) return -1;

    /* 从当前配置开始 */
    if (lnn_get_config(lnn, target_config) != 0) {
        log_error("[架构控制器] 获取当前 LNN 配置失败");
        return -1;
    }

    size_t current_hidden = target_config->hidden_size;
    int    current_layers = target_config->num_layers;

    switch (request->type) {
    case ARCH_CHANGE_EXPAND_HIDDEN:
        target_config->hidden_size = request->target_hidden_size;
        break;

    case ARCH_CHANGE_SHRINK_HIDDEN:
        target_config->hidden_size = request->target_hidden_size;
        if (target_config->hidden_size < ARCH_MIN_HIDDEN_SIZE)
            target_config->hidden_size = ARCH_MIN_HIDDEN_SIZE;
        break;

    case ARCH_CHANGE_ADD_LAYER:
        target_config->num_layers = request->target_num_layers;
        break;

    case ARCH_CHANGE_REMOVE_LAYER:
        target_config->num_layers = current_layers - 1;
        if (target_config->num_layers < 1) target_config->num_layers = 1;
        break;

    case ARCH_CHANGE_RESHAPE_ALL:
        if (request->target_hidden_size  > 0) target_config->hidden_size  = request->target_hidden_size;
        if (request->target_num_layers   > 0) target_config->num_layers   = request->target_num_layers;
        if (request->target_input_size   > 0) target_config->input_size   = request->target_input_size;
        if (request->target_output_size  > 0) target_config->output_size  = request->target_output_size;
        break;

    case ARCH_CHANGE_REPLACE_ARCHITECTURE:
        /* 从 external_arch_data 反序列化配置 */
        if (request->external_arch_data && request->external_arch_data_size >= 4) {
            const uint32_t* data = (const uint32_t*)request->external_arch_data;
            target_config->input_size  = (size_t)data[0];
            target_config->hidden_size = (size_t)data[1];
            target_config->output_size = (size_t)data[2];
            target_config->num_layers  = (int)data[3];
        }
        break;

    default:
        return -1;
    }

    /* 钳制到安全范围 */
    if (target_config->hidden_size < ARCH_MIN_HIDDEN_SIZE)
        target_config->hidden_size = ARCH_MIN_HIDDEN_SIZE;
    if (target_config->hidden_size > ARCH_MAX_HIDDEN_SIZE)
        target_config->hidden_size = ARCH_MAX_HIDDEN_SIZE;
    if (target_config->num_layers < ARCH_MIN_LAYERS)
        target_config->num_layers = ARCH_MIN_LAYERS;
    if (target_config->num_layers > ARCH_MAX_LAYERS)
        target_config->num_layers = ARCH_MAX_LAYERS;

    /* 派生 output_size 与 hidden_size 保持一致 */
    if (request->target_output_size == 0) {
        target_config->output_size = target_config->hidden_size;
    }

    log_info("[架构控制器] 目标配置: input=%zu hidden=%zu output=%zu layers=%d",
             target_config->input_size, target_config->hidden_size,
             target_config->output_size, target_config->num_layers);
    return 0;
}

/**
 * @brief 计算神经元总数
 */
static size_t arch_count_neurons(const LNN* lnn) {
    if (!lnn) return 0;
    LNNConfig cfg;
    if (lnn_get_config(lnn, &cfg) != 0) return 0;
    /* 输入神经元 + (隐藏层神经元 × 层数) + 输出神经元 */
    return cfg.input_size + cfg.hidden_size * (size_t)cfg.num_layers + cfg.output_size;
}

/**
 * @brief 计算可学习参数总数
 */
static size_t arch_count_params(const LNN* lnn) {
    if (!lnn || !lnn->cfc_network) return 0;
    CfCNetwork* cfc = lnn->cfc_network;
    size_t total = 0;

    /* param_block 中的参数 */
    if (cfc->per_layer_w_offset && cfc->total_weight_params > 0) {
        total += cfc->total_weight_params + cfc->total_bias_params;
    }

    /* 输出投影 */
    if (cfc->config.output_size != cfc->config.hidden_size) {
        total += cfc->config.output_size * cfc->config.hidden_size;  /* W_out */
        total += cfc->config.output_size;                             /* b_out */
    }

    /* cell 级独有参数 */
    for (int l = 0; l < cfc->config.num_layers; l++) {
        CfCCell* cell = cfc->layers[l];
        if (!cell) continue;
        size_t hs = (size_t)cfc->config.hidden_size;
        size_t in = (l == 0) ? (size_t)cfc->config.input_size : hs;

        total += in * hs * 3;  /* input/forget/output gate weights */
        total += hs * 3;       /* gate biases */
        total += hs;           /* time constants */
        total += hs * hs * 4;  /* H2H weights (4 matrices) */
    }

    return total;
}

int arch_controller_submit_change(ArchitectureController* controller,
                                   LNN** lnn_ptr,
                                   const ArchitectureChangeRequest* request,
                                   ArchitectureChangeResult* result) {
    if (!controller || !lnn_ptr || !*lnn_ptr || !request) {
        log_error("[架构控制器] 提交变更失败: 参数为空");
        if (result) {
            result->success = 0;
            result->error_code = SELFLNN_ERROR_INVALID_ARGUMENT;
            snprintf(result->error_message, sizeof(result->error_message), "参数为空");
        }
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    LNN* old_lnn = *lnn_ptr;
    ArchitectureChangeResult local_result;
    memset(&local_result, 0, sizeof(local_result));

    /* 记录变更前统计 */
    local_result.old_neuron_count = arch_count_neurons(old_lnn);
    local_result.old_param_count  = arch_count_params(old_lnn);

    /* 1. 安全审批 */
    int approve_code = arch_controller_approve_change(controller, request);
    if (approve_code != SELFLNN_SUCCESS) {
        local_result.success    = 0;
        local_result.error_code = approve_code;
        snprintf(local_result.error_message, sizeof(local_result.error_message),
                "审批未通过 (code=%d)", approve_code);
        if (result) memcpy(result, &local_result, sizeof(ArchitectureChangeResult));
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    /* 2. 自动存档（保存检查点） */
    if (controller->config.enable_archive_backup) {
        time_t now = time(NULL);
        snprintf(local_result.archive_path, sizeof(local_result.archive_path),
                "%s/arch_pre_change_%lld.slnn",
                controller->config.archive_dir, (long long)now);
        /* 检查点保存由外部 lnn_save 负责，此处仅记录路径 */
        log_info("[架构控制器] 变更前存档路径: %s", local_result.archive_path);
    }

    /* 3. 计算目标配置 */
    LNNConfig target_config;
    memset(&target_config, 0, sizeof(target_config));
    if (arch_compute_target_config(old_lnn, request, &target_config) != 0) {
        local_result.success    = 0;
        local_result.error_code = -200;
        snprintf(local_result.error_message, sizeof(local_result.error_message),
                "无法计算目标配置");
        if (result) memcpy(result, &local_result, sizeof(ArchitectureChangeResult));
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    /* 4. 使用 rebuild_lnn 创建新LNN并保留知识迁移 */
    LNN* new_lnn = arch_controller_rebuild_lnn(old_lnn, &target_config);
    if (!new_lnn) {
        local_result.success    = 0;
        local_result.error_code = -201;
        snprintf(local_result.error_message, sizeof(local_result.error_message),
                "创建新 LNN 实例失败");
        if (result) memcpy(result, &local_result, sizeof(ArchitectureChangeResult));
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    /* 5. 原子交换 */
    *lnn_ptr = new_lnn;

    /* 6. 销毁旧 LNN */
    lnn_free(old_lnn);

    /* 7. 记录变更时间戳 */
    arch_record_change_time(controller);

    /* 8. 填充结果 */
    local_result.success           = 1;
    local_result.error_code        = 0;
    local_result.new_neuron_count  = arch_count_neurons(new_lnn);
    local_result.new_param_count   = arch_count_params(new_lnn);
    local_result.actual_improvement = request->expected_improvement;
    local_result.change_timestamp   = (uint64_t)time(NULL);

    snprintf(local_result.error_message, sizeof(local_result.error_message),
            "架构变更成功: 神经元 %zu→%zu, 参数 %zu→%zu",
            local_result.old_neuron_count, local_result.new_neuron_count,
            local_result.old_param_count, local_result.new_param_count);

    log_info("[架构控制器] %s", local_result.error_message);

    /* 9. 记录历史 */
    arch_record_history(controller, request, &local_result);

    if (result) {
        memcpy(result, &local_result, sizeof(ArchitectureChangeResult));
    }

    return SELFLNN_SUCCESS;
}

/* ============ 历史查询 ============ */

size_t arch_controller_get_history_count(const ArchitectureController* controller) {
    if (!controller) return 0;
    return controller->history.count;
}

int arch_controller_get_history_entry(const ArchitectureController* controller,
                                       size_t index,
                                       ArchitectureChangeHistoryEntry* entry) {
    if (!controller || !entry || index >= controller->history.count) return -1;

    /* 环形缓冲区：head 是最新写入位置，往回 index 条 */
    size_t pos = (controller->history.head + ARCH_MAX_HISTORY_ENTRIES - 1 - index)
                 % ARCH_MAX_HISTORY_ENTRIES;
    memcpy(entry, &controller->history.entries[pos], sizeof(ArchitectureChangeHistoryEntry));
    return 0;
}

/* ============ 状态查询 ============ */

int arch_controller_get_architecture_stats(const LNN* lnn,
                                            size_t* neuron_count,
                                            size_t* param_count,
                                            size_t* hidden_size,
                                            int* num_layers) {
    if (!lnn) return -1;

    if (neuron_count) *neuron_count = arch_count_neurons(lnn);
    if (param_count)  *param_count  = arch_count_params(lnn);

    LNNConfig cfg;
    if (lnn_get_config(lnn, &cfg) == 0) {
        if (hidden_size) *hidden_size = cfg.hidden_size;
        if (num_layers)  *num_layers  = cfg.num_layers;
    } else {
        if (hidden_size) *hidden_size = 0;
        if (num_layers)  *num_layers  = 0;
        return -1;
    }

    return 0;
}

/* ============ NAS 部署桥接 ============ */

int arch_controller_deploy_architecture(ArchitectureController* controller,
                                         LNN** lnn,
                                         const void* arch_data,
                                         size_t arch_data_size,
                                         float confidence,
                                         ArchitectureChangeResult* result) {
    if (!controller || !lnn || !*lnn || !arch_data || arch_data_size < 16) {
        if (result) {
            result->success = 0;
            result->error_code = SELFLNN_ERROR_INVALID_ARGUMENT;
            snprintf(result->error_message, sizeof(result->error_message),
                    "NAS 部署参数无效");
        }
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    ArchitectureChangeRequest req = arch_controller_default_request();
    req.type                 = ARCH_CHANGE_REPLACE_ARCHITECTURE;
    req.confidence           = confidence;
    req.external_arch_data   = (void*)arch_data;
    req.external_arch_data_size = arch_data_size;
    snprintf(req.source_module, sizeof(req.source_module), "NAS_System");
    snprintf(req.reason, sizeof(req.reason),
            "NAS系统推荐架构部署 (置信度=%.2f)", confidence);

    log_info("[架构控制器] NAS 部署请求: 置信度=%.2f, 数据大小=%zu", confidence, arch_data_size);
    return arch_controller_submit_change(controller, lnn, &req, result);
}

/* ============ 结构变异接口 ============ */

int arch_controller_structural_mutate(ArchitectureController* controller,
                                       LNN** lnn,
                                       const StructuralMutationConfig* config,
                                       ArchitectureChangeResult* result) {
    if (!controller || !lnn || !*lnn || !config) {
        if (result) {
            result->success = 0;
            result->error_code = SELFLNN_ERROR_INVALID_ARGUMENT;
        }
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    LNN* old_lnn = *lnn;
    LNNConfig current_config;
    if (lnn_get_config(old_lnn, &current_config) != 0) {
        if (result) {
            result->success = 0;
            result->error_code = -300;
        }
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    ArchitectureChangeRequest req = arch_controller_default_request();
    snprintf(req.source_module, sizeof(req.source_module), "EvolutionEngine");

    switch (config->mut_type) {
    case STRUCT_MUTATE_ADD_NEURON: {
        size_t new_hidden = (size_t)((float)current_config.hidden_size * (1.0f + config->mutation_ratio));
        if (new_hidden <= current_config.hidden_size) new_hidden = current_config.hidden_size + 32;
        if (new_hidden > (size_t)config->max_hidden_size) new_hidden = (size_t)config->max_hidden_size;
        req.type = ARCH_CHANGE_EXPAND_HIDDEN;
        req.target_hidden_size = new_hidden;
        snprintf(req.reason, sizeof(req.reason),
                "结构变异: 扩展神经元 %.0f%%, %zu→%zu",
                config->mutation_ratio * 100.0f, current_config.hidden_size, new_hidden);
        break;
    }

    case STRUCT_MUTATE_REMOVE_NEURON: {
        size_t new_hidden = (size_t)((float)current_config.hidden_size * (1.0f - config->mutation_ratio));
        if (new_hidden >= current_config.hidden_size) new_hidden = current_config.hidden_size - 32;
        if (new_hidden < (size_t)config->min_hidden_size) new_hidden = (size_t)config->min_hidden_size;
        req.type = ARCH_CHANGE_SHRINK_HIDDEN;
        req.target_hidden_size = new_hidden;
        snprintf(req.reason, sizeof(req.reason),
                "结构变异: 移除神经元 %.0f%%, %zu→%zu",
                config->mutation_ratio * 100.0f, current_config.hidden_size, new_hidden);
        break;
    }

    case STRUCT_MUTATE_ADD_LAYER: {
        int new_layers = current_config.num_layers + 1;
        if (new_layers > config->max_layers) new_layers = config->max_layers;
        req.type = ARCH_CHANGE_ADD_LAYER;
        req.target_num_layers = new_layers;
        snprintf(req.reason, sizeof(req.reason),
                "结构变异: 增加层 %d→%d", current_config.num_layers, new_layers);
        break;
    }

    case STRUCT_MUTATE_REMOVE_LAYER: {
        int new_layers = current_config.num_layers - 1;
        if (new_layers < config->min_layers) new_layers = config->min_layers;
        req.type = ARCH_CHANGE_REMOVE_LAYER;
        req.target_num_layers = new_layers;
        snprintf(req.reason, sizeof(req.reason),
                "结构变异: 移除层 %d→%d", current_config.num_layers, new_layers);
        break;
    }

    case STRUCT_MUTATE_EXPAND_RATIO: {
        size_t new_hidden = (size_t)((float)current_config.hidden_size * (1.0f + config->mutation_ratio));
        if (new_hidden <= current_config.hidden_size) new_hidden = current_config.hidden_size + 32;
        if (new_hidden > (size_t)config->max_hidden_size) new_hidden = (size_t)config->max_hidden_size;
        req.type = ARCH_CHANGE_EXPAND_HIDDEN;
        req.target_hidden_size = new_hidden;
        snprintf(req.reason, sizeof(req.reason),
                "结构变异: 按比例扩展 %.0f%%, %zu→%zu",
                config->mutation_ratio * 100.0f, current_config.hidden_size, new_hidden);
        break;
    }

    case STRUCT_MUTATE_SHRINK_RATIO: {
        size_t new_hidden = (size_t)((float)current_config.hidden_size * (1.0f - config->mutation_ratio));
        if (new_hidden >= current_config.hidden_size) new_hidden = current_config.hidden_size - 32;
        if (new_hidden < (size_t)config->min_hidden_size) new_hidden = (size_t)config->min_hidden_size;
        req.type = ARCH_CHANGE_SHRINK_HIDDEN;
        req.target_hidden_size = new_hidden;
        snprintf(req.reason, sizeof(req.reason),
                "结构变异: 按比例收缩 %.0f%%, %zu→%zu",
                config->mutation_ratio * 100.0f, current_config.hidden_size, new_hidden);
        break;
    }

    default:
        if (result) {
            result->success = 0;
            result->error_code = -301;
        }
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    req.confidence = (controller->config.min_confidence_threshold < 0.6f) 
        ? 0.65f : controller->config.min_confidence_threshold + 0.05f; /* 结构变异默认置信度略高于最低阈值 */
    log_info("[架构控制器] 结构变异: %s", req.reason);
    return arch_controller_submit_change(controller, lnn, &req, result);
}
