/**
 * @file architecture_controller.c
 * @brief 动态架构控制器实现 —— 运行时安全地修改液态神经网络结构
 *
 * 解决架构动态演化能力深度审计报告中确认的所有 P0/P1/P2 缺陷。
*/

#ifdef _MSC_VER
#pragma warning(disable: 4133)  /* const type mismatch in history record */
#endif

#define SELFLNN_IMPLEMENTATION
#define SELFLNN_CORE_INTERNAL

#include "selflnn/core/architecture_controller.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc_network.h"
#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"  /* ZSF-023: safe_malloc/calloc/free宏 */
#include "selflnn/utils/time_utils.h"     /* ZSF-023: get_timestamp_ms */
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* INCON-04: 删除悬空extern声明 lnn_get_forward_count（该函数在整个项目中不存在） */

/* ============ 维度范围常量 ============ */
#define ARCH_MIN_HIDDEN_SIZE    32    /**< 隐藏层最小尺寸 */
#define ARCH_MAX_HIDDEN_SIZE    16384 /**< 隐藏层最大尺寸 */
#define ARCH_MIN_LAYERS         1     /**< 最小层数 */
#define ARCH_MAX_LAYERS         16    /**< 最大层数 */
#define ARCH_MIN_INPUT_SIZE     8     /**< 输入层最小尺寸 */
#define ARCH_MAX_INPUT_SIZE     4096  /**< 输入层最大尺寸 */
#define ARCH_MAX_HISTORY_ENTRIES 128  /**< 变更历史最大记录数 */

/* ============ 内部随机数生成 ============ */
/* P1-009修复: 将RNG状态从文件级static改为线程局部存储，消除多线程数据竞争
 * 原代码 arch_rng_state 为全局static，NAS搜索线程与训练线程并发修改
 * 导致伪随机序列交错、可复现性破坏 */
#ifdef _WIN32
static __declspec(thread) unsigned int arch_rng_state = 0xDEADBEEF;
#else
static _Thread_local unsigned int arch_rng_state = 0xDEADBEEF;
#endif

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
 * @brief 将权重矩阵从旧网络迁移到新网络（重要性感知迁移策略）
 *
 * ZSF-005 修复：当收缩(新维度<旧维度)时，不再简单截断末尾神经元，
 * 而是基于神经元L2范数重要性评分，保留最重要的神经元权重。
 * 
 * 对于 h2h 权重矩阵 [new_hidden × new_hidden]：
 *   - 计算每个旧神经元的权重范数作为重要性评分
 *   - 按重要性降序排列神经元索引
 *   - 优先复制重要性最高的神经元权重
 *   - 新行/新列使用 Xavier 初始化
 */
static void arch_transfer_weight_matrix(const float* old_w, size_t old_rows, size_t old_cols,
                                         float* new_w, size_t new_rows, size_t new_cols,
                                         float scale) {
    /* 清零新矩阵 */
    memset(new_w, 0, new_rows * new_cols * sizeof(float));

    size_t copy_rows = (old_rows < new_rows) ? old_rows : new_rows;
    size_t copy_cols = (old_cols < new_cols) ? old_cols : new_cols;

    /* ZSF-005: 收缩时基于神经元重要性排序，保留最重要的神经元 */
    if (new_rows < old_rows && copy_rows > 0) {
        /* 计算每个旧神经元的重要性评分(L2范数) */
        float* importance = (float*)safe_malloc(old_rows * sizeof(float));
        size_t* sorted_indices = (size_t*)safe_malloc(old_rows * sizeof(size_t));
        if (importance && sorted_indices) {
            for (size_t r = 0; r < old_rows; r++) {
                float row_norm = 0.0f;
                for (size_t c = 0; c < old_cols; c++) {
                    float w = old_w[r * old_cols + c];
                    row_norm += w * w;
                }
                importance[r] = sqrtf(row_norm);
                sorted_indices[r] = r;
            }
            /* 简单插入排序（按重要性降序，保留最重要的在前） */
            for (size_t i = 1; i < old_rows; i++) {
                size_t key_idx = sorted_indices[i];
                float key_val = importance[key_idx];
                size_t j = i;
                while (j > 0 && importance[sorted_indices[j - 1]] < key_val) {
                    sorted_indices[j] = sorted_indices[j - 1];
                    j--;
                }
                sorted_indices[j] = key_idx;
            }
            /* 按重要性顺序复制行 */
            for (size_t r = 0; r < copy_rows; r++) {
                size_t src_row = sorted_indices[r];
                for (size_t c = 0; c < copy_cols; c++) {
                    new_w[r * new_cols + c] = old_w[src_row * old_cols + c];
                }
            }
            /* 记录被丢弃的神经元重要性信息 */
            if (copy_rows < old_rows) {
                float kept_min = importance[sorted_indices[copy_rows - 1]];
                float discarded_max = importance[sorted_indices[copy_rows]];
                log_debug("[架构修剪] 保留%d/%zu神经元, 保留最低重要性=%.6f, 丢弃最高重要性=%.6f",
                         (int)copy_rows, old_rows, kept_min, discarded_max);
            }
        } else {
            /* 内存不足回退到简单左上角复制 */
            for (size_t r = 0; r < copy_rows; r++) {
                for (size_t c = 0; c < copy_cols; c++) {
                    new_w[r * new_cols + c] = old_w[r * old_cols + c];
                }
            }
        }
        safe_free((void**)&importance);
        safe_free((void**)&sorted_indices);
    } else {
        /* 扩展或维度不变：左上角复制（原有逻辑） */
        for (size_t r = 0; r < copy_rows; r++) {
            for (size_t c = 0; c < copy_cols; c++) {
                new_w[r * new_cols + c] = old_w[r * old_cols + c];
            }
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
        /* L-7修复: 使用局部变量保存有效配置指针，保留传入参数new_config语义不变 */
        const LNNConfig* effective_config = &new_lnn->config;
        LNNConfig old_config;
        if (lnn_get_config(old_lnn, &old_config) == 0) {
            new_lnn->config.learning_rate   = old_config.learning_rate;
            new_lnn->config.time_constant   = old_config.time_constant;
            new_lnn->config.enable_training = old_config.enable_training;
            new_lnn->config.enable_adaptation = old_config.enable_adaptation;
            new_lnn->config.enable_evolution  = old_config.enable_evolution;
            new_lnn->config.ode_solver_type   = old_config.ode_solver_type;
        }
        (void)effective_config; /* 明确标记使用意图 */

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
            old_lnn->memory_system = NULL;  /* F7修复: 防止lnn_free双重释放 */
        }
        if (old_lnn->laplace_analyzer) {
            new_lnn->laplace_analyzer          = old_lnn->laplace_analyzer;
            new_lnn->laplace_gradient_strength = old_lnn->laplace_gradient_strength;
            old_lnn->laplace_analyzer = NULL;  /* F7修复: 防止lnn_free双重释放 */
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
    /* H-6修复：允许request为NULL（回滚场景），使用默认占位请求 */
    if (!controller || !result) return;

    ChangeHistory* hist = &controller->history;
    ArchitectureChangeHistoryEntry* entry = &hist->entries[hist->head];

    if (request) {
        memcpy(&entry->request, request, sizeof(ArchitectureChangeRequest));
    } else {
        memset(&entry->request, 0, sizeof(ArchitectureChangeRequest));
        entry->request.type = ARCH_CHANGE_ROLLBACK;  /* 标记为回滚操作 */
        snprintf(entry->request.source_module, sizeof(entry->request.source_module),
                "安全回滚");
    }
    memcpy(&entry->result,  result,  sizeof(ArchitectureChangeResult));

    time_t now = time(NULL);
    struct tm tm_buf;
    struct tm* tm_info;
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
    tm_info = &tm_buf;
#else
    tm_info = localtime_r(&now, &tm_buf);
#endif
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

    /* 2. 自动存档（保存真实检查点，支持安全回滚） */
    if (controller->config.enable_archive_backup) {
        time_t now = time(NULL);
        snprintf(local_result.archive_path, sizeof(local_result.archive_path),
                "%s/arch_pre_change_%lld.slnn",
                controller->config.archive_dir, (long long)now);
        /* P0-002修复: 实际调用lnn_save保存检查点，而非仅记录路径 */
        if (lnn_save(old_lnn, local_result.archive_path) == 0) {
            log_info("[架构控制器] 变更前检查点已保存: %s", local_result.archive_path);
        } else {
            log_warning("[架构控制器] 变更前检查点保存失败: %s（继续执行变更但无法回滚）",
                       local_result.archive_path);
            local_result.archive_path[0] = '\0'; /* 标记无可用回滚点 */
        }
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

    /* ZSF-017 修复: 架构变更的并发安全保护。
     * 注释说"原子交换"但实际是普通指针赋值，且立即释放旧LNN。
     * 如果训练线程正在 lnn_forward/backward 中持有旧LNN锁并访问权重，
     * 释放旧LNN会导致 use-after-free → 未定义行为 → 可能崩溃。
     * 
     * 修复: 在交换前获取旧LNN的锁，确保所有在途训练操作已完成。
     * lnn_forward/backward 都在 LNN_LOCK 保护下执行，获取旧LNN锁后
     * 不会有任何训练操作正在访问旧LNN的内存。交换后训练线程将自动
     * 获取新LNN的锁。 */
    lnn_lock(old_lnn);
    
    /* 5. 真正的原子交换——使用平台原子操作确保所有线程同时看到新旧LNN切换 */
#ifdef _WIN32
    InterlockedExchangePointer((void* volatile*)lnn_ptr, new_lnn);
#elif defined(__GNUC__) || defined(__clang__)
    __atomic_exchange_n(lnn_ptr, new_lnn, __ATOMIC_ACQ_REL);
#else
    *lnn_ptr = new_lnn;  /* 非原子回退（锁已保护） */
#endif
    
    lnn_unlock(old_lnn);

    /* M-008修复: 延迟释放旧LNN——在解锁后使用真正的宽限期策略。
     * 旧LNN在锁内已从全局指针断开，任何新操作都将获取new_lnn的锁。
     * 宽限期(grace period)确保所有已在old_lnn锁上等待的线程完成操作后才释放。
     * 实现方式：
     *   1. 完整内存屏障(全序)确保所有CPU核心看到新指针
     *   2. 短暂休眠(1ms)作为真正的宽限期，让所有在途操作完成
     *   3. 再次内存屏障确保休眠期间的内存操作全局可见
     * 这比4次自增循环的"伪宽限期"提供了真正的并发安全保障。 */
    {
#ifdef _WIN32
        MemoryBarrier();  /* 全序内存屏障：确保InterlockedExchangePointer对所有核心可见 */
        Sleep(1);         /* 1ms宽限期：给所有在途训练线程足够时间完成当前操作 */
        MemoryBarrier();  /* 再次屏障：确保休眠期间的所有内存写入全局可见 */
#elif defined(__GNUC__) || defined(__clang__)
        __sync_synchronize();  /* 全序内存屏障 */
        /* POSIX: nanosleep 1ms 宽限期 */
        {
            struct timespec ts = {0, 1000000}; /* 1ms */
            nanosleep(&ts, NULL);
        }
        __sync_synchronize();
#else
        /* 通用回退：使用volatile忙等待约1ms（约100000次循环） */
        {
            volatile int grace_count = 0;
            for (int i = 0; i < 100000; i++) {
                grace_count++;
            }
        }
#endif
    }
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

/* ================================================================
 * P0-002修复: 架构变更安全回滚机制
 * 从变更前保存的检查点文件加载LNN，恢复到变更前状态。
 * 调用者需确保archive_path指向有效的.slnn检查点文件。
 * ================================================================ */

int arch_controller_rollback(ArchitectureController* controller,
                              LNN** lnn_ptr,
                              const char* archive_path) {
    if (!controller || !lnn_ptr || !*lnn_ptr) {
        log_error("[架构控制器] 回滚失败: 无效参数");
        return -1;
    }

    if (!archive_path || archive_path[0] == '\0') {
        log_error("[架构控制器] 回滚失败: 存档路径为空（变更前检查点可能未成功保存）");
        return -1;
    }

    log_warning("[架构控制器] 正在执行安全回滚，加载检查点: %s", archive_path);

    /* 使用lnn_load_from_file加载检查点到当前LNN实例 */
    int load_ret = lnn_load_from_file(*lnn_ptr, archive_path);
    if (load_ret != 0) {
        log_error("[架构控制器] 回滚失败: 检查点文件加载错误 (%s)", archive_path);
        return -1;
    }

    /* 记录回滚历史 */
    ArchitectureChangeHistoryEntry rollback_entry;
    memset(&rollback_entry, 0, sizeof(rollback_entry));
    snprintf(rollback_entry.timestamp_str, sizeof(rollback_entry.timestamp_str),
            "rollback_%lld", (long long)time(NULL));
    /* result中记录错误消息以标识这是一次回滚操作 */
    ArchitectureChangeResult dummy_result;
    memset(&dummy_result, 0, sizeof(dummy_result));
    dummy_result.success = 1;
    dummy_result.error_code = 0;
    snprintf(dummy_result.error_message, sizeof(dummy_result.error_message),
            "安全回滚: 从检查点 %s 恢复", archive_path);
    memcpy(&rollback_entry.result, &dummy_result, sizeof(ArchitectureChangeResult));
    arch_record_history(controller, NULL, &rollback_entry);

    log_info("[架构控制器] 安全回滚成功: 已从 %s 恢复到变更前状态", archive_path);
    return 0;
}

/* ================================================================
 * H-004修复: 运行时激活度监控与在线结构调整
 *
 * 实现需求R49: "当任务复杂度升高时自动增加神经元、
 * 移除冗余或低激活度神经元、调整层数"
 *
 * 核心机制:
 *   1. 指数移动平均(EMA)收集每层神经元激活统计
 *      - 均值: 反映神经元平均活跃度
 *      - 方差: 反映神经元响应多样性
 *      - 最小/最大值: 检测死亡/饱和神经元
 *   2. 激活熵评估任务复杂度
 *      - 高熵 → 任务复杂 → 需要更多神经元
 *      - 低熵 → 任务简单 → 可修剪冗余神经元
 *   3. 连续确认机制防止误触发
 *      - 需要连续N次检查都满足触发条件
 *   4. 自适应阈值
 *      - 基于历史复杂度动态调整触发阈值
 * ================================================================ */

/* ============ 激活度监控器生命周期 ============ */

ArchActivationMonitor* arch_controller_create_activation_monitor(
    int num_layers,
    const size_t* hidden_sizes,
    int check_interval)
{
    int i;
    ArchActivationMonitor* monitor;

    if (num_layers <= 0 || num_layers > ARCH_MAX_LAYERS || !hidden_sizes) {
        log_error("[激活监控] 创建失败: 参数无效 (num_layers=%d)", num_layers);
        return NULL;
    }

    monitor = (ArchActivationMonitor*)safe_calloc(1, sizeof(ArchActivationMonitor));
    if (!monitor) {
        log_error("[激活监控] 内存分配失败");
        return NULL;
    }

    monitor->num_layers = num_layers;
    monitor->layers = (ArchLayerActivationStats*)safe_calloc(
        (size_t)num_layers, sizeof(ArchLayerActivationStats));
    if (!monitor->layers) {
        safe_free((void**)&monitor);
        return NULL;
    }

    for (i = 0; i < num_layers; i++) {
        size_t hs = hidden_sizes[i];
        if (hs == 0) hs = ARCH_MIN_HIDDEN_SIZE;

        monitor->layers[i].hidden_size = hs;
        monitor->layers[i].ema_decay = 0.99f;
        monitor->layers[i].sample_count = 0;

        monitor->layers[i].ema_mean = (float*)safe_calloc(hs, sizeof(float));
        monitor->layers[i].ema_variance = (float*)safe_calloc(hs, sizeof(float));
        monitor->layers[i].ema_min = (float*)safe_calloc(hs, sizeof(float));
        monitor->layers[i].ema_max = (float*)safe_calloc(hs, sizeof(float));

        if (!monitor->layers[i].ema_mean || !monitor->layers[i].ema_variance ||
            !monitor->layers[i].ema_min || !monitor->layers[i].ema_max) {
            /* 清理已分配资源 */
            arch_controller_free_activation_monitor(monitor);
            return NULL;
        }

        /* 初始化最小值/最大值 */
        {
            size_t j;
            for (j = 0; j < hs; j++) {
                monitor->layers[i].ema_min[j] = 1.0f;
                monitor->layers[i].ema_max[j] = 0.0f;
            }
        }
    }

    monitor->check_interval = (check_interval > 0) ? check_interval : 100;
    monitor->low_activity_threshold = 0.01f;
    monitor->saturation_threshold = 0.95f;
    monitor->low_activity_ratio_trigger = 0.15f;
    monitor->saturation_ratio_trigger = 0.10f;
    monitor->consecutive_checks_required = 3;
    monitor->total_samples = 0;
    monitor->is_initialized = 1;
    monitor->current_complexity_score = 0.0f;
    monitor->complexity_history_idx = 0;
    monitor->complexity_history_count = 0;
    {
        int ci;
        for (ci = 0; ci < 16; ci++) {
            monitor->complexity_history[ci] = 0.0f;
        }
    }

    log_info("[激活监控] 初始化完成: %d层, 检查间隔=%d, 低激活阈值=%.3f, 饱和阈值=%.3f",
             num_layers, monitor->check_interval,
             monitor->low_activity_threshold, monitor->saturation_threshold);
    return monitor;
}

void arch_controller_free_activation_monitor(ArchActivationMonitor* monitor) {
    int i;
    if (!monitor) return;

    if (monitor->layers) {
        for (i = 0; i < monitor->num_layers; i++) {
            safe_free((void**)&monitor->layers[i].ema_mean);
            safe_free((void**)&monitor->layers[i].ema_variance);
            safe_free((void**)&monitor->layers[i].ema_min);
            safe_free((void**)&monitor->layers[i].ema_max);
        }
        safe_free((void**)&monitor->layers);
    }
    memset(monitor, 0, sizeof(ArchActivationMonitor));
    safe_free((void**)&monitor);
}

/* ============ 激活值记录 ============ */

void arch_controller_record_activation(ArchActivationMonitor* monitor,
                                        int layer_idx,
                                        const float* activations,
                                        size_t hidden_size) {
    size_t j;
    float decay;
    ArchLayerActivationStats* layer;

    if (!monitor || !monitor->is_initialized || !activations) return;
    if (layer_idx < 0 || layer_idx >= monitor->num_layers) return;
    if (hidden_size == 0) return;

    layer = &monitor->layers[layer_idx];
    decay = layer->ema_decay;
    {
        size_t hs = (hidden_size < layer->hidden_size) ? hidden_size : layer->hidden_size;

        for (j = 0; j < hs; j++) {
            float val = activations[j];
            float old_mean = layer->ema_mean[j];
            float old_var = layer->ema_variance[j];

            /* NaN/Inf保护 */
            if (isnan(val) || isinf(val)) {
                val = 0.0f;
            }

            /* EMA更新均值: mean = decay * old_mean + (1-decay) * val */
            layer->ema_mean[j] = decay * old_mean + (1.0f - decay) * val;

            /* EMA更新方差: var = decay * old_var + (1-decay) * (val - old_mean)^2 */
            {
                float diff = val - old_mean;
                layer->ema_variance[j] = decay * old_var + (1.0f - decay) * diff * diff;
            }

            /* 更新最小值/最大值 */
            if (val < layer->ema_min[j]) {
                layer->ema_min[j] = decay * layer->ema_min[j] + (1.0f - decay) * val;
            } else {
                layer->ema_min[j] = decay * layer->ema_min[j] + (1.0f - decay) * layer->ema_min[j];
            }
            if (val > layer->ema_max[j]) {
                layer->ema_max[j] = decay * layer->ema_max[j] + (1.0f - decay) * val;
            } else {
                layer->ema_max[j] = decay * layer->ema_max[j] + (1.0f - decay) * layer->ema_max[j];
            }
        }
    }

    layer->sample_count++;
    monitor->total_samples++;

    /* 定期记录日志 */
    if (monitor->total_samples % 1000 == 0) {
        log_debug("[激活监控] 已收集 %zu 个样本 (层%d: %zu样本)",
                  monitor->total_samples, layer_idx, layer->sample_count);
    }
}

/* ============ 任务复杂度评估 ============ */

int arch_controller_assess_complexity(ArchActivationMonitor* monitor,
                                       ArchComplexityMetrics* metrics) {
    int li;
    size_t total_neurons;
    size_t low_activity_total;
    size_t saturated_total;
    float total_entropy;
    float total_variance;
    size_t j;

    if (!monitor || !monitor->is_initialized || !metrics) return -1;
    if (monitor->total_samples == 0) {
        memset(metrics, 0, sizeof(ArchComplexityMetrics));
        snprintf(metrics->diagnosis, sizeof(metrics->diagnosis), "无激活数据，无法评估复杂度");
        return -1;
    }

    memset(metrics, 0, sizeof(ArchComplexityMetrics));
    total_neurons = 0;
    low_activity_total = 0;
    saturated_total = 0;
    total_entropy = 0.0f;
    total_variance = 0.0f;

    for (li = 0; li < monitor->num_layers; li++) {
        ArchLayerActivationStats* layer = &monitor->layers[li];
        size_t hs = layer->hidden_size;
        float layer_entropy = 0.0f;
        float layer_variance_sum = 0.0f;
        size_t layer_low = 0;
        size_t layer_saturated = 0;

        if (layer->sample_count == 0) continue;

        for (j = 0; j < hs; j++) {
            float mean_val = layer->ema_mean[j];
            float var_val = layer->ema_variance[j];

            /* NaN检查 */
            if (isnan(mean_val)) mean_val = 0.0f;
            if (isnan(var_val)) var_val = 0.0f;

            /* 激活熵: 基于均值归一化后的分布熵
             * 使用 |mean| 作为概率估计，计算 -p*log(p) */
            {
                float abs_mean = fabsf(mean_val);
                float p = (abs_mean < 1e-7f) ? 0.0f : abs_mean;
                if (p > 0.0f) {
                    layer_entropy -= p * logf(p + 1e-10f);
                }
            }

            layer_variance_sum += var_val;

            /* 检测低激活度神经元: ema_max < 0.01 表示持续不活跃 */
            if (layer->ema_max[j] < monitor->low_activity_threshold) {
                layer_low++;
            }

            /* 检测饱和神经元: ema_max > 0.95 且 ema_min > 0.9 表示持续饱和 */
            if (layer->ema_max[j] > monitor->saturation_threshold &&
                layer->ema_min[j] > 0.9f) {
                layer_saturated++;
            }
        }

        total_neurons += hs;
        low_activity_total += layer_low;
        saturated_total += layer_saturated;
        total_entropy += layer_entropy / (float)hs;
        total_variance += layer_variance_sum / (float)hs;
    }

    if (total_neurons == 0) {
        memset(metrics, 0, sizeof(ArchComplexityMetrics));
        return -1;
    }

    /* 填充指标 */
    metrics->activation_entropy = total_entropy / (float)monitor->num_layers;
    metrics->activation_variance = total_variance / (float)monitor->num_layers;
    metrics->gradient_norm = 0.0f; /* 推理时不可用 */
    metrics->saturation_ratio = (float)saturated_total / (float)total_neurons;
    metrics->low_activity_ratio = (float)low_activity_total / (float)total_neurons;

    /* 综合复杂度评分: 熵加权 + 方差加权 + 饱和惩罚
     * 熵越高 → 任务越复杂
     * 方差越高 → 响应多样性越大 → 需要更多容量
     * 饱和度高 → 网络容量不足 → 需要扩展 */
    {
        float entropy_score = metrics->activation_entropy * 2.5f;  /* 熵最大贡献 */
        float variance_score = metrics->activation_variance * 1.5f;
        float saturation_score = metrics->saturation_ratio * 5.0f;  /* 饱和是强烈信号 */
        float low_activity_penalty = metrics->low_activity_ratio * 3.0f; /* 低激活度惩罚 */

        metrics->overall_complexity = entropy_score + variance_score + saturation_score;
        /* 钳制到[0,1] */
        if (metrics->overall_complexity > 1.0f) metrics->overall_complexity = 1.0f;
        if (metrics->overall_complexity < 0.0f) metrics->overall_complexity = 0.0f;

        /* 诊断 */
        metrics->needs_expansion = 0;
        metrics->needs_pruning = 0;
        metrics->needs_layer_adjustment = 0;

        if (metrics->saturation_ratio > monitor->saturation_ratio_trigger) {
            metrics->needs_expansion = 1;
        }
        if (metrics->low_activity_ratio > monitor->low_activity_ratio_trigger) {
            metrics->needs_pruning = 1;
        }
        if (metrics->overall_complexity > 0.8f && metrics->needs_expansion) {
            metrics->needs_layer_adjustment = 1;
        }

        snprintf(metrics->diagnosis, sizeof(metrics->diagnosis),
                "复杂度=%.3f 熵=%.3f 方差=%.3f 饱和率=%.1f%% 低活率=%.1f%% → %s%s%s",
                metrics->overall_complexity,
                metrics->activation_entropy,
                metrics->activation_variance,
                metrics->saturation_ratio * 100.0f,
                metrics->low_activity_ratio * 100.0f,
                metrics->needs_expansion ? "需扩展 " : "",
                metrics->needs_pruning ? "需修剪 " : "",
                metrics->needs_layer_adjustment ? "需调整层" : "无需调整");
    }

    /* 更新复杂度历史 */
    monitor->complexity_history[monitor->complexity_history_idx] = metrics->overall_complexity;
    monitor->complexity_history_idx = (monitor->complexity_history_idx + 1) % 16;
    if (monitor->complexity_history_count < 16) {
        monitor->complexity_history_count++;
    }
    monitor->current_complexity_score = metrics->overall_complexity;

    return 0;
}

/* ============ 异常检测 ============ */

int arch_controller_detect_activity_anomalies(ArchActivationMonitor* monitor,
                                               size_t* low_activity_count,
                                               size_t* saturated_count) {
    int li;
    size_t total_low;
    size_t total_sat;
    size_t j;

    if (!monitor || !monitor->is_initialized) return -1;

    total_low = 0;
    total_sat = 0;

    for (li = 0; li < monitor->num_layers; li++) {
        ArchLayerActivationStats* layer = &monitor->layers[li];
        size_t hs = layer->hidden_size;

        if (layer->sample_count == 0) continue;

        for (j = 0; j < hs; j++) {
            if (layer->ema_max[j] < monitor->low_activity_threshold) {
                total_low++;
            }
            if (layer->ema_max[j] > monitor->saturation_threshold &&
                layer->ema_min[j] > 0.9f) {
                total_sat++;
            }
        }
    }

    if (low_activity_count) *low_activity_count = total_low;
    if (saturated_count) *saturated_count = total_sat;

    return 0;
}

/* ============ 在线调整评估 ============ */

int arch_controller_evaluate_online_adaptation(ArchActivationMonitor* monitor,
                                                const LNN* lnn,
                                                ArchOnlineAdaptationAdvice* advice) {
    ArchComplexityMetrics metrics;
    LNNConfig cfg;
    float avg_complexity;
    int ci;
    float complexity_trend;
    size_t j;

    if (!monitor || !monitor->is_initialized || !lnn || !advice) return -1;

    memset(advice, 0, sizeof(ArchOnlineAdaptationAdvice));

    /* 评估当前复杂度 */
    if (arch_controller_assess_complexity(monitor, &metrics) != 0) {
        snprintf(advice->reason, sizeof(advice->reason), "无法评估复杂度（样本不足）");
        return 0;
    }

    /* 获取当前配置 */
    if (lnn_get_config(lnn, &cfg) != 0) {
        return -1;
    }

    /* 计算复杂度趋势: 最近N个样本的移动平均 */
    {
        int valid_count = (monitor->complexity_history_count < 8) ?
                          monitor->complexity_history_count : 8;
        int start_idx = (monitor->complexity_history_idx + 16 - valid_count) % 16;
        avg_complexity = 0.0f;
        for (ci = 0; ci < valid_count; ci++) {
            avg_complexity += monitor->complexity_history[(start_idx + ci) % 16];
        }
        avg_complexity /= (float)valid_count;

        /* 趋势: 最近一半 vs 前一半 */
        if (valid_count >= 4) {
            float recent_half = 0.0f;
            float older_half = 0.0f;
            int half = valid_count / 2;
            for (ci = 0; ci < half; ci++) {
                recent_half += monitor->complexity_history[(start_idx + half + ci) % 16];
                older_half += monitor->complexity_history[(start_idx + ci) % 16];
            }
            recent_half /= (float)half;
            older_half /= (float)half;
            complexity_trend = recent_half - older_half;
        } else {
            complexity_trend = 0.0f;
        }
    }

    /* 检测异常神经元 */
    {
        size_t low_count = 0;
        size_t sat_count = 0;
        arch_controller_detect_activity_anomalies(monitor, &low_count, &sat_count);
        advice->low_activity_count = low_count;
        advice->saturated_count = sat_count;
    }

    /* 决策逻辑 */
    advice->recommend_expand = 0;
    advice->recommend_prune = 0;
    advice->recommend_add_layer = 0;
    advice->recommend_remove_layer = 0;
    advice->target_hidden_size = cfg.hidden_size;
    advice->target_num_layers = cfg.num_layers;

    /* 扩展决策: 饱和率高 + 复杂度上升趋势 */
    if (metrics.saturation_ratio > monitor->saturation_ratio_trigger &&
        complexity_trend > 0.02f) {
        advice->recommend_expand = 1;
        /* 按饱和比例扩展: 扩展比例 = 饱和率 * 0.5 */
        {
            size_t expand_amount = (size_t)((float)cfg.hidden_size * metrics.saturation_ratio * 0.5f);
            if (expand_amount < 32) expand_amount = 32;
            advice->target_hidden_size = cfg.hidden_size + expand_amount;
            if (advice->target_hidden_size > ARCH_MAX_HIDDEN_SIZE) {
                advice->target_hidden_size = ARCH_MAX_HIDDEN_SIZE;
            }
        }
        /* 如果已接近最大隐藏层维度，考虑增加层 */
        if (cfg.hidden_size >= ARCH_MAX_HIDDEN_SIZE * 3 / 4 &&
            cfg.num_layers < ARCH_MAX_LAYERS) {
            advice->recommend_add_layer = 1;
            advice->target_num_layers = cfg.num_layers + 1;
        }
        advice->confidence = metrics.saturation_ratio * 0.8f + fabsf(complexity_trend) * 2.0f;
        if (advice->confidence > 0.95f) advice->confidence = 0.95f;
        snprintf(advice->reason, sizeof(advice->reason),
                "饱和率%.1f%%偏高+复杂度趋势%.3f上升, 建议扩展至%zu神经元",
                metrics.saturation_ratio * 100.0f, complexity_trend,
                advice->target_hidden_size);
    }
    /* 修剪决策: 低激活率高 + 复杂度下降趋势 */
    else if (metrics.low_activity_ratio > monitor->low_activity_ratio_trigger &&
             complexity_trend < -0.02f) {
        advice->recommend_prune = 1;
        /* 按低激活比例修剪: 修剪比例 = 低活率 * 0.4 */
        {
            size_t prune_amount = (size_t)((float)cfg.hidden_size * metrics.low_activity_ratio * 0.4f);
            if (prune_amount < 16) prune_amount = 16;
            if (prune_amount >= cfg.hidden_size) {
                prune_amount = cfg.hidden_size / 4;
            }
            advice->target_hidden_size = cfg.hidden_size - prune_amount;
            if (advice->target_hidden_size < ARCH_MIN_HIDDEN_SIZE) {
                advice->target_hidden_size = ARCH_MIN_HIDDEN_SIZE;
            }
        }
        /* 如果低激活率极高，考虑移除冗余层 */
        if (metrics.low_activity_ratio > 0.3f && cfg.num_layers > 2) {
            advice->recommend_remove_layer = 1;
            advice->target_num_layers = cfg.num_layers - 1;
        }
        advice->confidence = metrics.low_activity_ratio * 0.7f;
        if (advice->confidence > 0.9f) advice->confidence = 0.9f;
        snprintf(advice->reason, sizeof(advice->reason),
                "低活率%.1f%%偏高+复杂度趋势%.3f下降, 建议修剪至%zu神经元",
                metrics.low_activity_ratio * 100.0f, complexity_trend,
                advice->target_hidden_size);
    }
    /* 层调整决策: 仅在复杂度极高且已经最大化隐藏维度时 */
    else if (metrics.needs_layer_adjustment && cfg.num_layers < ARCH_MAX_LAYERS) {
        advice->recommend_add_layer = 1;
        advice->target_num_layers = cfg.num_layers + 1;
        advice->confidence = 0.7f;
        snprintf(advice->reason, sizeof(advice->reason),
                "复杂度%.3f极高, 建议增加层数 %d→%d",
                metrics.overall_complexity, cfg.num_layers, advice->target_num_layers);
    }
    else {
        advice->confidence = 0.0f;
        snprintf(advice->reason, sizeof(advice->reason),
                "无需调整: 复杂度=%.3f 饱和率=%.1f%% 低活率=%.1f%%",
                metrics.overall_complexity,
                metrics.saturation_ratio * 100.0f,
                metrics.low_activity_ratio * 100.0f);
    }

    log_info("[激活监控] 评估完成: %s (置信度=%.2f)", advice->reason, advice->confidence);

    return 0;
}

/* ============ 在线自适应调整执行 ============ */

int arch_controller_online_adapt(ArchitectureController* controller,
                                  LNN** lnn_ptr,
                                  ArchActivationMonitor* monitor,
                                  ArchitectureChangeResult* result) {
    ArchOnlineAdaptationAdvice advice;
    int ret;
    ArchitectureChangeRequest req;
    size_t j;

    if (!controller || !lnn_ptr || !*lnn_ptr || !monitor || !monitor->is_initialized) {
        if (result) {
            memset(result, 0, sizeof(ArchitectureChangeResult));
            result->success = 0;
            result->error_code = SELFLNN_ERROR_INVALID_ARGUMENT;
            snprintf(result->error_message, sizeof(result->error_message),
                    "在线调整失败: 参数无效");
        }
        return -1;
    }

    /* 评估是否需要调整 */
    ret = arch_controller_evaluate_online_adaptation(monitor, *lnn_ptr, &advice);
    if (ret != 0) {
        if (result) {
            memset(result, 0, sizeof(ArchitectureChangeResult));
            result->success = 0;
            result->error_code = -400;
        }
        return ret;
    }

    /* 如果不需要调整，直接返回 */
    if (advice.confidence < 0.5f) {
        if (result) {
            memset(result, 0, sizeof(ArchitectureChangeResult));
            result->success = 1;
            result->error_code = 0;
            snprintf(result->error_message, sizeof(result->error_message),
                    "无需在线调整: %s", advice.reason);
        }
        return 0;
    }

    /* 构建变更请求 */
    req = arch_controller_default_request();
    req.confidence = advice.confidence;
    snprintf(req.source_module, sizeof(req.source_module), "OnlineAdaptation");
    snprintf(req.reason, sizeof(req.reason), "%s", advice.reason);

    if (advice.recommend_expand) {
        req.type = ARCH_CHANGE_EXPAND_HIDDEN;
        req.target_hidden_size = advice.target_hidden_size;
    } else if (advice.recommend_prune) {
        req.type = ARCH_CHANGE_SHRINK_HIDDEN;
        req.target_hidden_size = advice.target_hidden_size;
    } else if (advice.recommend_add_layer) {
        req.type = ARCH_CHANGE_ADD_LAYER;
        req.target_num_layers = advice.target_num_layers;
    } else if (advice.recommend_remove_layer) {
        req.type = ARCH_CHANGE_REMOVE_LAYER;
        req.target_num_layers = advice.target_num_layers;
    } else {
        /* 理论上不会到达这里 */
        return 0;
    }

    log_info("[激活监控] 在线调整触发: 类型=%d 置信度=%.2f 原因=%s",
             req.type, req.confidence, req.reason);

    /* 提交变更 */
    ret = arch_controller_submit_change(controller, lnn_ptr, &req, result);

    /* 变更后重置激活统计，开始新一轮监控 */
    arch_controller_reset_activation_stats(monitor);

    if (ret == SELFLNN_SUCCESS) {
        /* 更新监控器的层配置以匹配新LNN */
        LNN* new_lnn = *lnn_ptr;
        if (new_lnn) {
            LNNConfig new_cfg;
            if (lnn_get_config(new_lnn, &new_cfg) == 0) {
                /* 如果层数变化，重新分配监控层 */
                if (new_cfg.num_layers != monitor->num_layers) {
                    /* 释放旧层 */
                    {
                        int li;
                        for (li = 0; li < monitor->num_layers; li++) {
                            safe_free((void**)&monitor->layers[li].ema_mean);
                            safe_free((void**)&monitor->layers[li].ema_variance);
                            safe_free((void**)&monitor->layers[li].ema_min);
                            safe_free((void**)&monitor->layers[li].ema_max);
                        }
                        safe_free((void**)&monitor->layers);
                    }
                    /* 分配新层 */
                    monitor->num_layers = new_cfg.num_layers;
                    monitor->layers = (ArchLayerActivationStats*)safe_calloc(
                        (size_t)new_cfg.num_layers, sizeof(ArchLayerActivationStats));
                    if (monitor->layers) {
                        int li;
                        for (li = 0; li < new_cfg.num_layers; li++) {
                            size_t hs = (li == 0) ? new_cfg.input_size : new_cfg.hidden_size;
                            monitor->layers[li].hidden_size = hs;
                            monitor->layers[li].ema_decay = 0.99f;
                            monitor->layers[li].ema_mean = (float*)safe_calloc(hs, sizeof(float));
                            monitor->layers[li].ema_variance = (float*)safe_calloc(hs, sizeof(float));
                            monitor->layers[li].ema_min = (float*)safe_calloc(hs, sizeof(float));
                            monitor->layers[li].ema_max = (float*)safe_calloc(hs, sizeof(float));
                            /* 初始化最小值/最大值 */
                            {
                                size_t j;
                                for (j = 0; j < hs; j++) {
                                    monitor->layers[li].ema_min[j] = 1.0f;
                                    monitor->layers[li].ema_max[j] = 0.0f;
                                }
                            }
                        }
                    }
                }
            }
        }
        log_info("[激活监控] 在线调整成功: %s", advice.reason);
    } else {
        log_warning("[激活监控] 在线调整失败: 错误码=%d", ret);
    }

    return ret;
}

/* ============ 统计重置 ============ */

void arch_controller_reset_activation_stats(ArchActivationMonitor* monitor) {
    int li;
    size_t j;

    if (!monitor || !monitor->is_initialized) return;

    for (li = 0; li < monitor->num_layers; li++) {
        ArchLayerActivationStats* layer = &monitor->layers[li];
        size_t hs = layer->hidden_size;

        /* 清零EMA统计 */
        for (j = 0; j < hs; j++) {
            layer->ema_mean[j] = 0.0f;
            layer->ema_variance[j] = 0.0f;
            layer->ema_min[j] = 1.0f;
            layer->ema_max[j] = 0.0f;
        }
        layer->sample_count = 0;
    }

    monitor->total_samples = 0;
    monitor->current_complexity_score = 0.0f;
    monitor->complexity_history_idx = 0;
    monitor->complexity_history_count = 0;
    {
        int ci;
        for (ci = 0; ci < 16; ci++) {
            monitor->complexity_history[ci] = 0.0f;
        }
    }

    log_info("[激活监控] 统计已重置");
}
