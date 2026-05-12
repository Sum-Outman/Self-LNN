/**
 * @file auto_learning.h
 * @brief 自主学习知识库系统接口
 */

#ifndef SELFLNN_AUTO_LEARNING_H
#define SELFLNN_AUTO_LEARNING_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 学习模式 */
typedef enum {
    AUTO_LEARN_MODE_WATCH = 0,       /* 监控模式：检测文件变化并自动学习 */
    AUTO_LEARN_MODE_BATCH = 1,       /* 批量模式：一次性学习所有文件 */
    AUTO_LEARN_MODE_INCREMENTAL = 2, /* 增量模式：增量更新已有知识 */
    AUTO_LEARN_MODE_INTERACTIVE = 3  /* 交互模式：用户确认后再添加 */
} AutoLearnMode;

/* 知识源类型 */
typedef enum {
    KNOWLEDGE_SOURCE_TEXT = 0,       /* 纯文本 */
    KNOWLEDGE_SOURCE_MARKDOWN = 1,   /* Markdown */
    KNOWLEDGE_SOURCE_JSON = 2,       /* JSON */
    KNOWLEDGE_SOURCE_CSV = 3,        /* CSV表格 */
    KNOWLEDGE_SOURCE_CODE = 4,       /* 代码文件 */
    KNOWLEDGE_SOURCE_MANUAL = 5,     /* 说明书/手册 */
    KNOWLEDGE_SOURCE_DIALOGUE = 6    /* 对话记录 */
} KnowledgeSourceType;

/* 学习条目 */
typedef struct {
    char* topic;                     /* 主题 */
    char* content;                   /* 内容 */
    KnowledgeSourceType source_type; /* 来源类型 */
    char source_path[512];           /* 来源文件路径 */
    time_t learned_at;               /* 学习时间 */
    float confidence;                /* 置信度 */
    int verified;                    /* 是否已验证 */
    char* extracted_entities[16];    /* 抽取的实体 */
    int entity_count;                /* 实体数量 */
    char* extracted_relations[16];   /* 抽取的关系 */
    int relation_count;              /* 关系数量 */
} AutoLearnEntry;

/* 知识融合模式 */
typedef enum {
    AUTO_LEARN_FUSE_MERGE = 0,       /* 合并模式：合并所有来源的知识 */
    AUTO_LEARN_FUSE_OVERRIDE = 1,    /* 覆盖模式：高置信度覆盖低置信度 */
    AUTO_LEARN_FUSE_AVERAGE = 2      /* 平均模式：置信度加权平均 */
} AutoLearnFusionMode;

/* 融合结果记录 */
typedef struct {
    int source_entry_count;           /* 参与融合的源条目数 */
    int source_indices[32];           /* 源条目索引 */
    int target_entry_index;           /* 融合后的目标条目索引 */
    float confidence_before;          /* 融合前置信度 */
    float confidence_after;           /* 融合后置信度 */
    time_t fusion_time;               /* 融合时间 */
    char topic[256];                  /* 融合主题 */
} AutoLearnFusionResult;

/* 学习统计 */
typedef struct {
    size_t total_files_scanned;      /* 扫描的文件数 */
    size_t total_entries_learned;     /* 学习的条目数 */
    size_t total_entities_extracted;  /* 抽取的实体数 */
    size_t total_relations_extracted; /* 抽取的关系数 */
    size_t conflicts_detected;        /* 检测到的冲突数 */
    size_t conflicts_resolved;        /* 解决的冲突数 */
    time_t last_scan_time;            /* 最后扫描时间 */
    time_t total_scan_time_ms;        /* 总扫描时间（毫秒） */
    float average_confidence;         /* 平均置信度 */
} AutoLearnStats;

/* 自主学习系统句柄 */
typedef struct AutoLearningSystem AutoLearningSystem;

/**
 * @brief 创建自主学习系统
 */
AutoLearningSystem* auto_learning_create(AutoLearnMode mode);

/**
 * @brief 释放自主学习系统
 */
void auto_learning_free(AutoLearningSystem* system);

/**
 * @brief 设置知识库目录
 */
int auto_learning_set_directory(AutoLearningSystem* system, const char* directory);

/**
 * @brief 扫描并学习目录中的所有文件
 */
int auto_learning_scan_directory(AutoLearningSystem* system);

/**
 * @brief 学习单个文件
 */
int auto_learning_learn_file(AutoLearningSystem* system, const char* filepath);

/**
 * @brief 从文本内容学习
 */
int auto_learning_learn_text(AutoLearningSystem* system, const char* text, 
                            const char* source, KnowledgeSourceType type);

/**
 * @brief 验证已学习的知识
 */
int auto_learning_verify_knowledge(AutoLearningSystem* system, size_t entry_index);

/**
 * @brief 检测新知识与已有知识的冲突
 */
int auto_learning_detect_conflicts(AutoLearningSystem* system, 
                                   int* conflict_indices, int max_conflicts);

/**
 * @brief 解决知识冲突
 */
int auto_learning_resolve_conflict(AutoLearningSystem* system, int conflict_index, 
                                   int keep_new);

/**
 * @brief 开始文件监控
 */
int auto_learning_start_watching(AutoLearningSystem* system);

/**
 * @brief 停止文件监控
 */
int auto_learning_stop_watching(AutoLearningSystem* system);

/**
 * @brief 获取学习统计
 */
int auto_learning_get_stats(const AutoLearningSystem* system, AutoLearnStats* stats);

/**
 * @brief 获取学习条目数量
 */
size_t auto_learning_get_entry_count(const AutoLearningSystem* system);

/**
 * @brief 获取学习条目
 */
const AutoLearnEntry* auto_learning_get_entry(const AutoLearningSystem* system, size_t index);

/**
 * @brief 导出学习结果到知识库
 */
int auto_learning_export_to_knowledge_base(AutoLearningSystem* system, void* knowledge_base);

/**
 * @brief 增量更新：学习新内容时检查相似已有条目，合并更新而非新建
 * @param system 自主学习系统
 * @param topic 主题
 * @param content 内容
 * @param type 来源类型
 * @param source_path 来源路径
 * @return 成功返回更新的条目索引，失败返回-1
 */
int auto_learning_incremental_update(AutoLearningSystem* system, const char* topic,
                                     const char* content, KnowledgeSourceType type,
                                     const char* source_path);

/**
 * @brief 知识融合：将多个关于同一话题的条目融合为一条综合知识
 * @param system 自主学习系统
 * @param topic 融合的主题
 * @param source_indices 源条目索引数组
 * @param count 源条目数量
 * @param mode 融合模式
 * @param result 融合结果输出（可为NULL）
 * @return 成功返回0，失败返回-1
 */
int auto_learning_fuse_knowledge(AutoLearningSystem* system, const char* topic,
                                  int source_indices[], int count,
                                  AutoLearnFusionMode mode,
                                  AutoLearnFusionResult* result);

/**
 * @brief 置信度动态更新：基于交叉引用、来源可靠度、时效性动态调整置信度
 * @param system 自主学习系统
 * @param entry_index 条目索引
 * @return 更新后的置信度（0-1），失败返回-1
 */
float auto_learning_update_confidence(AutoLearningSystem* system, size_t entry_index);

/**
 * @brief 自动重学习：扫描过期知识，重新读取来源文件并更新
 * @param system 自主学习系统
 * @param max_days 超过多少天未更新视为过期
 * @return 成功重学习的条目数，失败返回-1
 */
int auto_learning_relearn_expired(AutoLearningSystem* system, int max_days);

/* =========================================================================
 * A03.4.1 主动学习系统
 * ========================================================================= */

/** 不确定性采样策略 */
typedef enum {
    AL_UNCERTAINTY_LEAST_CONFIDENCE = 0,  /* 最低置信度采样 */
    AL_UNCERTAINTY_MARGIN = 1,            /* 边界采样（最大两概率差值最小） */
    AL_UNCERTAINTY_ENTROPY = 2            /* 熵最大采样 */
} ALUncertaintyStrategy;

/** 主动学习样本 */
typedef struct {
    float* features;              /* 特征向量 */
    int feature_dim;              /* 特征维度 */
    float uncertainty_score;      /* 不确定性分数 */
    float diversity_score;        /* 多样性分数 */
    float combined_score;         /* 综合分数 */
    int label;                    /* 标签（-1为未标注） */
    int selected;                 /* 是否已选中 */
    char* metadata;               /* 元数据 */
} ALSample;

/** 主动学习器句柄 */
typedef struct ALActiveLearner ALActiveLearner;

/**
 * @brief 创建主动学习器
 * @param uncertainty_strategy 不确定性采样策略
 * @return 主动学习器句柄，失败返回NULL
 */
ALActiveLearner* active_learner_create(ALUncertaintyStrategy uncertainty_strategy);

/**
 * @brief 释放主动学习器
 */
void active_learner_free(ALActiveLearner* learner);

/**
 * @brief 添加样本到池
 * @param learner 主动学习器
 * @param features 特征向量
 * @param feature_dim 特征维度
 * @param label 标签（-1为未标注）
 * @param metadata 元数据（可选）
 * @return 样本索引，失败返回-1
 */
int active_learner_add_sample(ALActiveLearner* learner, const float* features,
                               int feature_dim, int label, const char* metadata);

/**
 * @brief 批量添加样本
 * @param learner 主动学习器
 * @param features 特征数组 [sample_count][feature_dim]
 * @param feature_dim 特征维度
 * @param labels 标签数组
 * @param sample_count 样本数量
 * @return 成功添加的样本数
 */
int active_learner_add_samples(ALActiveLearner* learner, const float* features,
                                int feature_dim, const int* labels, int sample_count);

/**
 * @brief 获取样本总数
 */
int active_learner_sample_count(const ALActiveLearner* learner);

/**
 * @brief 获取已标注样本数
 */
int active_learner_labeled_count(const ALActiveLearner* learner);

/**
 * @brief 不确定性采样：选择最不确定的N个样本
 * @param learner 主动学习器
 * @param strategy 不确定性策略（覆盖创建时的设置）
 * @param sample_indices 输出选中的样本索引数组
 * @param n 需要选择的样本数
 * @return 实际选择的样本数
 */
int active_learner_uncertainty_sampling(ALActiveLearner* learner,
    ALUncertaintyStrategy strategy, int* sample_indices, int n);

/**
 * @brief 多样性采样：基于K-means聚类的多样性选择
 * @param learner 主动学习器
 * @param sample_indices 输出选中的样本索引数组
 * @param n 需要选择的样本数
 * @param cluster_count 聚类数（0表示自动估算）
 * @return 实际选择的样本数
 */
int active_learner_diversity_sampling(ALActiveLearner* learner,
    int* sample_indices, int n, int cluster_count);

/**
 * @brief 查询合成：合成位于决策边界附近的样本
 * @param learner 主动学习器
 * @param out_features 输出合成的特征向量
 * @param feature_dim 特征维度
 * @param count 需要合成的样本数
 * @return 实际合成的样本数
 */
int active_learner_query_synthesis(ALActiveLearner* learner,
    float* out_features, int feature_dim, int count);

/**
 * @brief 综合查询策略：不确定性+多样性排序选择
 * @param learner 主动学习器
 * @param sample_indices 输出选中的样本索引数组
 * @param n 需要选择的样本数
 * @param uncertainty_weight 不确定性权重
 * @param diversity_weight 多样性权重
 * @return 实际选择的样本数
 */
int active_learner_query(ALActiveLearner* learner, int* sample_indices, int n,
                          float uncertainty_weight, float diversity_weight);

/**
 * @brief 标记样本标签
 * @param learner 主动学习器
 * @param sample_index 样本索引
 * @param label 标签
 * @return 成功返回0，失败返回-1
 */
int active_learner_label_sample(ALActiveLearner* learner, int sample_index, int label);

/**
 * @brief 计算所有未标注样本的不确定性分数
 * @param learner 主动学习器
 * @param strategy 采样策略
 * @return 成功返回0，失败返回-1
 */
int active_learner_compute_uncertainties(ALActiveLearner* learner,
    ALUncertaintyStrategy strategy);

/**
 * @brief 获取样本的详细信息
 * @param learner 主动学习器
 * @param index 样本索引
 * @return ALSample* 样本指针，失败返回NULL
 */
const ALSample* active_learner_get_sample(const ALActiveLearner* learner, int index);

/* =========================================================================
 * A03.4.2 持续学习系统
 * ========================================================================= */

/** 弹性权重巩固（EWC）句柄 */
typedef struct EWC EWC;

/**
 * @brief 创建EWC（弹性权重巩固）
 * @param param_count 参数数量
 * @param lambda 正则化强度（建议100-10000）
 * @return EWC句柄，失败返回NULL
 */
EWC* ewc_create(int param_count, float lambda);

/**
 * @brief 释放EWC
 */
void ewc_free(EWC* ewc);

/**
 * @brief 设置当前任务的最优参数
 * @param ewc EWC句柄
 * @param optimal_params 最优参数数组
 * @return 成功返回0，失败返回-1
 */
int ewc_set_optimal_params(EWC* ewc, const float* optimal_params);

/**
 * @brief 计算Fisher信息矩阵（使用样本计算参数重要性）
 * @param ewc EWC句柄
 * @param gradients 梯度样本数组 [sample_count][param_count]
 * @param sample_count 样本数量
 * @return 成功返回0，失败返回-1
 */
int ewc_compute_fisher(EWC* ewc, const float* gradients, int sample_count);

/**
 * @brief 计算EWC正则化损失（对新参数的惩罚项）
 * @param ewc EWC句柄
 * @param current_params 当前参数
 * @param out_loss 输出损失值
 * @param out_per_param_loss 输出每个参数的损失（可选，长度为param_count）
 * @return 成功返回0，失败返回-1
 */
int ewc_compute_loss(EWC* ewc, const float* current_params,
                     float* out_loss, float* out_per_param_loss);

/**
 * @brief 合并多个EWC实例（多任务联合巩固）
 * @param targets 目标EWC数组
 * @param count EWC数量
 * @return 合并后的EWC句柄，失败返回NULL
 */
EWC* ewc_merge(EWC** targets, int count);

/**
 * @brief 获取参数重要性（Fisher对角线值）
 * @param ewc EWC句柄
 * @param out_importance 输出重要性数组
 * @return 成功返回0，失败返回-1
 */
int ewc_get_importance(EWC* ewc, float* out_importance);

/** 渐进式神经网络句柄 */
typedef struct ProgressiveNet ProgressiveNet;

/** 网络列（Column）：每个任务一个列 */
typedef struct {
    int input_dim;                 /* 输入维度 */
    int hidden_dim;               /* 隐藏层维度 */
    int output_dim;               /* 输出维度 */
    float* w_in;                  /* 输入权重 [input_dim][hidden_dim] */
    float* b_in;                  /* 输入偏置 [hidden_dim] */
    float* w_hidden;              /* 隐藏权重 [hidden_dim][hidden_dim] */
    float* b_hidden;              /* 隐藏偏置 [hidden_dim] */
    float* w_out;                 /* 输出权重 [hidden_dim][output_dim] */
    float* b_out;                 /* 输出偏置 [output_dim] */
    int frozen;                   /* 是否冻结 */
} ProgressiveColumn;

/**
 * @brief 创建渐进式神经网络
 * @param input_dim 输入维度
 * @param hidden_dim 隐藏层维度
 * @return ProgressiveNet* 句柄，失败返回NULL
 */
ProgressiveNet* progressive_net_create(int input_dim, int hidden_dim);

/**
 * @brief 释放渐进式神经网络
 */
void progressive_net_free(ProgressiveNet* pnn);

/**
 * @brief 添加新任务列（冻结旧列，创建新列+横向连接）
 * @param pnn 渐进式网络
 * @param output_dim 新任务的输出维度
 * @return 新任务的列索引，失败返回-1
 */
int progressive_net_add_column(ProgressiveNet* pnn, int output_dim);

/**
 * @brief 获取列数量
 */
int progressive_net_column_count(const ProgressiveNet* pnn);

/**
 * @brief 获取指定列的权重
 * @param pnn 渐进式网络
 * @param column_idx 列索引
 * @return ProgressiveColumn* 列指针，失败返回NULL
 */
const ProgressiveColumn* progressive_net_get_column(const ProgressiveNet* pnn, int column_idx);

/**
 * @brief 前向传播（新列使用旧列的横向连接特征）
 * @param pnn 渐进式网络
 * @param column_idx 列索引
 * @param input 输入向量
 * @param output 输出向量
 * @return 成功返回0，失败返回-1
 */
int progressive_net_forward(ProgressiveNet* pnn, int column_idx,
                             const float* input, float* output);

/**
 * @brief 训练新列一步（仅更新新列的权重，冻结旧列）
 * @param pnn 渐进式网络
 * @param column_idx 列索引
 * @param input 输入向量
 * @param target 目标输出
 * @param learning_rate 学习率
 * @return 成功返回0，失败返回-1
 */
int progressive_net_train_step(ProgressiveNet* pnn, int column_idx,
                                const float* input, const float* target,
                                float learning_rate);

/** 记忆重放缓冲区句柄 */
typedef struct ExperienceReplay ExperienceReplay;

/** 经验样本 */
typedef struct {
    float* state;                  /* 状态/输入 */
    float* action;                 /* 动作/输出 */
    float reward;                  /* 奖励 */
    float* next_state;             /* 下一状态 */
    int done;                      /* 是否终止 */
    float priority;                /* 优先级 */
    int task_id;                   /* 任务ID */
} ExperienceSample;

/**
 * @brief 创建经验重放缓冲区
 * @param capacity 缓冲区容量
 * @param state_dim 状态维度
 * @param action_dim 动作维度
 * @return ExperienceReplay* 句柄，失败返回NULL
 */
ExperienceReplay* experience_replay_create(int capacity, int state_dim, int action_dim);

/**
 * @brief 释放经验重放缓冲区
 */
void experience_replay_free(ExperienceReplay* er);

/**
 * @brief 添加经验到缓冲区
 * @param er 经验重放
 * @param state 状态
 * @param action 动作
 * @param reward 奖励
 * @param next_state 下一状态
 * @param done 是否终止
 * @param task_id 任务ID
 * @return 成功返回0，失败返回-1
 */
int experience_replay_add(ExperienceReplay* er, const float* state,
                           const float* action, float reward,
                           const float* next_state, int done, int task_id);

/**
 * @brief 均匀采样一批经验
 * @param er 经验重放
 * @param batch_size 批次大小
 * @param out_states 输出状态数组
 * @param out_actions 输出动作数组
 * @param out_rewards 输出奖励数组
 * @param out_next_states 输出下一状态数组
 * @param out_dones 输出终止标志数组
 * @return 实际采样数量
 */
int experience_replay_sample(ExperienceReplay* er, int batch_size,
                              float* out_states, float* out_actions,
                              float* out_rewards, float* out_next_states,
                              int* out_dones);

/**
 * @brief 优先采样（基于优先级）
 * @param er 经验重放
 * @param batch_size 批次大小
 * @param beta 优先级采样指数（0=均匀，1=完全按优先级）
 * @param out_states 输出状态数组
 * @param out_actions 输出动作数组
 * @param out_rewards 输出奖励数组
 * @param out_next_states 输出下一状态数组
 * @param out_dones 输出终止标志数组
 * @param out_indices 输出采样索引
 * @param out_weights 输出重要性权重
 * @return 实际采样数量
 */
int experience_replay_priority_sample(ExperienceReplay* er, int batch_size,
    float beta, float* out_states, float* out_actions,
    float* out_rewards, float* out_next_states, int* out_dones,
    int* out_indices, float* out_weights);

/**
 * @brief 更新采样经验的优先级
 * @param er 经验重放
 * @param indices 索引数组
 * @param priorities 新优先级数组
 * @param count 数量
 * @return 成功返回0，失败返回-1
 */
int experience_replay_update_priorities(ExperienceReplay* er,
    const int* indices, const float* priorities, int count);

/**
 * @brief 获取缓冲区当前大小
 */
int experience_replay_size(const ExperienceReplay* er);

/**
 * @brief 获取缓冲区容量
 */
int experience_replay_capacity(const ExperienceReplay* er);

/**
 * @brief 清除缓冲区
 * @param er 经验重放
 */
void experience_replay_clear(ExperienceReplay* er);

/**
 * @brief 按任务ID获取缓冲区统计
 * @param er 经验重放
 * @param task_id 任务ID
 * @return 该任务的样本数
 */
int experience_replay_task_count(const ExperienceReplay* er, int task_id);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_AUTO_LEARNING_H */
