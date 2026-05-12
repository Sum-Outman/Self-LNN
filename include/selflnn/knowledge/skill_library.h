/**
 * @file skill_library.h
 * @brief 技能库系统接口
 * 
 * 技能库系统 - 存储和管理AGI可执行的任务技能。
 * 技能分层：原子技能→组合技能→任务技能。
 * 支持技能学习、检索、共享和执行。
 */

#ifndef SELFLNN_SKILL_LIBRARY_H
#define SELFLNN_SKILL_LIBRARY_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 技能库常量 */
#define SKILL_MAX_NAME 128
#define SKILL_MAX_DESC 512
#define SKILL_MAX_DOMAIN 64
#define SKILL_MAX_PREREQ 16
#define SKILL_MAX_STEPS 64
#define SKILL_MAX_PARAMS 16
#define SKILL_MAX_TAGS 32
#define SKILL_MAX_TAG_LEN 32
#define SKILL_MAX_TOOLS 16
#define SKILL_MAX_TOOL_LEN 64

/* 技能类型 */
typedef enum {
    SKILL_TYPE_ATOMIC = 0,       /* 原子技能：不可分割的基本操作 */
    SKILL_TYPE_COMPOSITE = 1,    /* 组合技能：多个原子技能的组合 */
    SKILL_TYPE_TASK = 2,         /* 任务技能：完整任务的端到端技能 */
    SKILL_TYPE_META = 3          /* 元技能：学习/生成其他技能的技能 */
} SkillType;

/* 技能执行状态 */
typedef enum {
    SKILL_STATUS_IDLE = 0,       /* 空闲 */
    SKILL_STATUS_RUNNING = 1,    /* 运行中 */
    SKILL_STATUS_COMPLETED = 2,  /* 已完成 */
    SKILL_STATUS_FAILED = 3,     /* 失败 */
    SKILL_STATUS_PAUSED = 4,     /* 暂停 */
    SKILL_STATUS_ABORTED = 5     /* 中止 */
} SkillStatus;

/* 参数类型 */
typedef enum {
    SKILL_PARAM_FLOAT = 0,       /* 浮点数 */
    SKILL_PARAM_INT = 1,         /* 整数 */
    SKILL_PARAM_STRING = 2,      /* 字符串 */
    SKILL_PARAM_BOOL = 3,        /* 布尔值 */
    SKILL_PARAM_VECTOR = 4,      /* 向量 */
    SKILL_PARAM_MATRIX = 5,      /* 矩阵 */
    SKILL_PARAM_OBJECT = 6       /* 对象引用 */
} SkillParamType;

/* 前置条件类型 */
typedef enum {
    SKILL_PREREQ_KNOWLEDGE = 0,   /* 知识前提 */
    SKILL_PREREQ_SKILL = 1,       /* 技能前提 */
    SKILL_PREREQ_RESOURCE = 2,    /* 资源前提 */
    SKILL_PREREQ_STATE = 3,       /* 状态前提 */
    SKILL_PREREQ_SENSOR = 4       /* 传感器前提 */
} SkillPrereqType;

/* 技能参数定义 */
typedef struct {
    char name[SKILL_MAX_NAME];           /* 参数名 */
    SkillParamType type;                 /* 参数类型 */
    char default_value[SKILL_MAX_DESC];  /* 默认值 */
    char description[SKILL_MAX_DESC];    /* 参数描述 */
    int required;                        /* 是否必需 */
    float min_value;                     /* 最小值（数值类型） */
    float max_value;                     /* 最大值（数值类型） */
} SkillParameter;

/* 前置条件 */
typedef struct {
    SkillPrereqType type;                /* 前置条件类型 */
    char condition[SKILL_MAX_DESC];      /* 条件描述 */
    char value[SKILL_MAX_NAME];          /* 条件值 */
    int satisfied;                       /* 是否已满足 */
} SkillPrerequisite;

/* 技能步骤 */
typedef struct {
    char action[SKILL_MAX_NAME];         /* 动作名称 */
    char description[SKILL_MAX_DESC];    /* 步骤描述 */
    char target[SKILL_MAX_NAME];         /* 目标对象 */
    char params[SKILL_MAX_DESC];         /* 参数 */
    float estimated_time_ms;             /* 估计时间（毫秒） */
    int retry_count;                     /* 重试次数 */
    int max_retries;                     /* 最大重试次数 */
} SkillStep;

/* 后置条件 */
typedef struct {
    char condition[SKILL_MAX_DESC];      /* 条件描述 */
    char expected_value[SKILL_MAX_NAME]; /* 期望值 */
    float tolerance;                     /* 容差 */
    int verified;                        /* 是否已验证 */
} SkillPostcondition;

/* 技能效果评估 */
typedef struct {
    float success_rate;                  /* 成功率 (0-1) */
    float average_duration_ms;           /* 平均执行时间 */
    float energy_cost;                   /* 能耗估计 */
    float quality_score;                 /* 质量评分 (0-1) */
    size_t execution_count;              /* 执行次数 */
    size_t success_count;                /* 成功次数 */
    time_t last_executed;                /* 最后执行时间 */
} SkillEffectiveness;

/* 技能记录 */
typedef struct {
    int skill_id;                        /* 技能唯一ID */
    char name[SKILL_MAX_NAME];           /* 技能名称 */
    char description[SKILL_MAX_DESC];    /* 技能描述 */
    SkillType type;                      /* 技能类型 */
    char domain[SKILL_MAX_DOMAIN];       /* 领域 */
    
    /* 参数定义 */
    SkillParameter params[SKILL_MAX_PARAMS];  /* 参数列表 */
    int param_count;                          /* 参数数量 */
    
    /* 前置条件 */
    SkillPrerequisite prerequisites[SKILL_MAX_PREREQ]; /* 前置条件列表 */
    int prerequisite_count;                           /* 前置条件数量 */
    
    /* 技能步骤 */
    SkillStep steps[SKILL_MAX_STEPS];    /* 步骤列表 */
    int step_count;                      /* 步骤数量 */
    
    /* 后置条件 */
    SkillPostcondition postconditions[SKILL_MAX_PREREQ]; /* 后置条件列表 */
    int postcondition_count;                           /* 后置条件数量 */
    
    /* 元信息 */
    char tags[SKILL_MAX_TAGS][SKILL_MAX_TAG_LEN]; /* 标签 */
    int tag_count;                                /* 标签数量 */
    char required_tools[SKILL_MAX_TOOLS][SKILL_MAX_TOOL_LEN]; /* 所需工具 */
    int tool_count;                                          /* 工具数量 */
    
    /* 效果评估 */
    SkillEffectiveness effectiveness;    /* 效果评估 */
    
    /* 技能关系 */
    int parent_skill_id;                 /* 父技能ID（如果是子技能） */
    int child_skill_ids[SKILL_MAX_PREREQ]; /* 子技能ID列表 */
    int child_count;                     /* 子技能数量 */
    
    /* 存储信息 */
    float embedding[64];                 /* 技能语义嵌入 */
    time_t created_at;                   /* 创建时间 */
    time_t updated_at;                   /* 更新时间 */
    int version;                         /* 版本号 */
    int enabled;                         /* 是否启用 */
    
    /* 执行上下文 */
    SkillStatus current_status;          /* 当前状态 */
    float progress;                      /* 执行进度 (0-1) */
} SkillRecord;

/* 技能库统计 */
typedef struct {
    size_t total_skills;                 /* 总技能数 */
    size_t atomic_skills;                /* 原子技能数 */
    size_t composite_skills;             /* 组合技能数 */
    size_t task_skills;                  /* 任务技能数 */
    size_t meta_skills;                  /* 元技能数 */
    float average_success_rate;          /* 平均成功率 */
    float total_executions;              /* 总执行次数 */
    size_t active_tasks;                 /* 当前活动任务数 */
    size_t skill_categories;             /* 技能类别数 */
} SkillLibraryStats;

/* 技能库句柄 */
typedef struct SkillLibrary SkillLibrary;

/**
 * @brief 创建技能库
 * @param max_skills 最大技能数（0表示无限制）
 * @return 技能库句柄，失败返回NULL
 */
SkillLibrary* skill_library_create(size_t max_skills);

/**
 * @brief 释放技能库
 * @param library 技能库句柄
 */
void skill_library_free(SkillLibrary* library);

/**
 * @brief 添加技能记录
 * @param library 技能库句柄
 * @param record 技能记录
 * @return 技能ID，失败返回-1
 */
int skill_library_add(SkillLibrary* library, const SkillRecord* record);

/**
 * @brief 更新技能记录
 * @param library 技能库句柄
 * @param skill_id 技能ID
 * @param record 新的技能记录
 * @return 0成功，-1失败
 */
int skill_library_update(SkillLibrary* library, int skill_id, const SkillRecord* record);

/**
 * @brief 删除技能
 * @param library 技能库句柄
 * @param skill_id 技能ID
 * @return 0成功，-1失败
 */
int skill_library_remove(SkillLibrary* library, int skill_id);

/**
 * @brief 获取技能记录
 * @param library 技能库句柄
 * @param skill_id 技能ID
 * @param record 输出技能记录
 * @return 0成功，-1失败
 */
int skill_library_get(const SkillLibrary* library, int skill_id, SkillRecord* record);

/**
 * @brief 按名称查找技能
 * @param library 技能库句柄
 * @param name 技能名称
 * @return 技能ID，未找到返回-1
 */
int skill_library_find_by_name(const SkillLibrary* library, const char* name);

/**
 * @brief 按标签搜索技能
 * @param library 技能库句柄
 * @param tag 标签
 * @param results 结果ID数组
 * @param max_results 最大结果数
 * @return 结果数量
 */
int skill_library_search_by_tag(const SkillLibrary* library, const char* tag, 
                                int* results, int max_results);

/**
 * @brief 按领域搜索技能
 * @param library 技能库句柄
 * @param domain 领域名
 * @param results 结果ID数组
 * @param max_results 最大结果数
 * @return 结果数量
 */
int skill_library_search_by_domain(const SkillLibrary* library, const char* domain,
                                   int* results, int max_results);

/**
 * @brief 语义搜索技能
 * @param library 技能库句柄
 * @param embedding 查询嵌入向量
 * @param dim 嵌入维度
 * @param results 结果ID数组
 * @param max_results 最大结果数
 * @return 结果数量
 */
int skill_library_search_semantic(const SkillLibrary* library, const float* embedding,
                                  int dim, int* results, int max_results);

/**
 * @brief 组合技能（从多个技能组合新技能）
 * @param library 技能库句柄
 * @param skill_ids 技能ID数组
 * @param count 技能数量
 * @param name 新技能名称
 * @param description 新技能描述
 * @return 新技能ID，失败返回-1
 */
int skill_library_compose(SkillLibrary* library, const int* skill_ids, int count,
                          const char* name, const char* description);

/**
 * @brief 从执行历史提取技能
 * @param library 技能库句柄
 * @param execution_log 执行日志（JSON格式字符串）
 * @return 新技能ID，失败返回-1
 */
int skill_library_extract_from_execution(SkillLibrary* library, const char* execution_log);

/**
 * @brief 评估技能执行前置条件
 * @param library 技能库句柄
 * @param skill_id 技能ID
 * @param context 执行上下文（当前状态）
 * @return 全部满足返回1，部分满足返回0，不满足返回-1
 */
int skill_library_check_prerequisites(const SkillLibrary* library, int skill_id,
                                      const void* context);

/**
 * @brief 更新技能效果评估
 * @param library 技能库句柄
 * @param skill_id 技能ID
 * @param success 是否成功
 * @param duration_ms 执行时间
 * @param quality 质量评分
 */
void skill_library_update_effectiveness(SkillLibrary* library, int skill_id,
                                        int success, float duration_ms, float quality);

/**
 * @brief 获取技能库统计
 * @param library 技能库句柄
 * @param stats 输出统计信息
 * @return 0成功，-1失败
 */
int skill_library_get_stats(const SkillLibrary* library, SkillLibraryStats* stats);

/**
 * @brief 获取所有技能ID
 * @param library 技能库句柄
 * @param ids 输出ID数组
 * @param max_ids 最大数量
 * @return 实际数量
 */
int skill_library_get_all_skills(const SkillLibrary* library, int* ids, int max_ids);

/**
 * @brief 技能库序列化（保存到文件）
 * @param library 技能库句柄
 * @param filepath 文件路径
 * @return 0成功，-1失败
 */
int skill_library_save(const SkillLibrary* library, const char* filepath);

/**
 * @brief 技能库反序列化（从文件加载）
 * @param library 技能库句柄
 * @param filepath 文件路径
 * @return 0成功，-1失败
 */
int skill_library_load(SkillLibrary* library, const char* filepath);

/**
 * @brief 获取技能间的依赖关系图
 * @param library 技能库句柄
 * @param skill_id 技能ID
 * @param dependency_ids 依赖技能ID数组
 * @param max_deps 最大数量
 * @return 依赖数量
 */
int skill_library_get_dependencies(const SkillLibrary* library, int skill_id,
                                   int* dependency_ids, int max_deps);

/**
 * @brief 获取技能的推荐下一步技能
 * @param library 技能库句柄
 * @param skill_id 技能ID
 * @param results 推荐技能ID数组
 * @param max_results 最大数量
 * @return 推荐数量
 */
int skill_library_get_next_skills(const SkillLibrary* library, int skill_id,
                                  int* results, int max_results);

/* ================================================================
 * K-021: 技能执行引擎
 * ================================================================ */

/**
 * @brief K-021: 技能执行结果
 */
typedef struct {
    int skill_id;
    char skill_name[128];
    int success;
    float duration_ms;
    float quality_score;
    char output_log[1024];
    int error_code;
    char error_message[256];
} SkillExecutionResult;

/**
 * @brief K-021: 执行指定技能
 *
 * 根据技能记录中的定义执行技能。
 * 支持：robot控制命令、programming代码片段、device操作、LNN推理、knowledge查询。
 * 执行前自动检查前置依赖，执行后记录质量评分。
 *
 * @param library 技能库句柄
 * @param skill_id 技能ID
 * @param params 参数JSON字符串(可NULL)
 * @param result 执行结果输出
 * @return 0成功，-1失败
 */
int skill_library_execute(SkillLibrary* library, int skill_id,
                           const char* params,
                           SkillExecutionResult* result);

/**
 * @brief K-021: 批量执行技能序列
 *
 * 按顺序执行技能ID数组，前一个技能输出作为后一个技能输入。
 * 任一技能失败则立即停止并返回失败。
 *
 * @param library 技能库句柄
 * @param skill_ids 技能ID数组
 * @param count 技能数量
 * @param results 执行结果数组[count]
 * @return 成功执行数，失败返回-1
 */
int skill_library_execute_sequence(SkillLibrary* library,
                                    const int* skill_ids, int count,
                                    SkillExecutionResult* results);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SKILL_LIBRARY_H */
