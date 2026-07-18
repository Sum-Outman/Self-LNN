/**
 * @file product_design.c
 * @brief 产品设计能力核心实现 —— 产品设计基础层
 * 
 * K-011: 角色定义 —— product_design.c 是产品设计的【核心算法层】
 * 负责：需求解析、规格生成、设计评估和优化。
 * 
 * 层级关系：
 *   product_design.c（本文件）→ 基础产品设计算法
 *   product_design_enhanced.c → 增强特性（多目标优化、约束求解、仿真验证）
 *   两者互补，通过产品设计子系统统一API暴露
 */

#include "selflnn/product_design/product_design.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/core/errors.h"
#include "selflnn/reasoning/reasoning.h"
#include "selflnn/core/laplace.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>

/**
 * @brief 知识库条目：产品领域知识
 */
typedef struct {
    char* domain;                /**< 领域名称 */
    char* description;           /**< 知识描述 */
    double cost_factor;          /**< 成本因子 */
    double time_factor;          /**< 时间因子 */
    double complexity_factor;    /**< 复杂度因子 */
    double quality_base;         /**< 基础质量评分 0-1 */
} ProductKnowledgeEntry;

/**
 * @brief 知识库
 */
typedef struct {
    ProductKnowledgeEntry* entries;  /**< 知识条目数组 */
    size_t count;                    /**< 条目数量 */
    size_t capacity;                 /**< 容量 */
} ProductKnowledgeBase;

/**
 * @brief 设计规则
 */
typedef struct {
    char* condition;             /**< 条件关键词 */
    char* action;                /**< 建议行动 */
    double weight;               /**< 规则权重 */
    int is_constraint;           /**< 是否为硬约束 */
    double impact_factor;        /**< 影响因子 */
} DesignRuleEntry;

/**
 * @brief 规则引擎
 */
typedef struct {
    DesignRuleEntry* rules;      /**< 规则数组 */
    size_t count;                /**< 规则数量 */
    size_t capacity;             /**< 容量 */
} DesignRuleEngine;

/**
 * @brief 评估模型参数
 */
typedef struct {
    double feasibility_weight;   /**< 可行性权重 */
    double cost_weight;          /**< 成本权重 */
    double innovation_weight;    /**< 创新权重 */
    double market_weight;        /**< 市场权重 */
    double quality_threshold;    /**< 质量阈值 */
    double cost_sensitivity;     /**< 成本敏感度 */
    double innovation_factor;    /**< 创新因子 */
} EvaluationModelParams;

/**
 * @brief 产品设计引擎内部结构体
 */
/* 参考设计案例条目 */
#define PDE_MAX_REFERENCE_CASES 128
typedef struct {
    char name[128];              /**< 案例名称 */
    char category_name[32];      /**< 类别名称 */
    char feature_tags[512];      /**< 关键特征标签 */
    char constraint_summary[512]; /**< 设计约束摘要 */
    int category;                /**< 类别枚举值 */
} ProductReferenceCase;

struct ProductDesignEngine {
    int is_initialized;          /**< 是否已初始化 */
    ProductKnowledgeBase* knowledge_base;  /**< 产品知识库 */
    DesignRuleEngine* rule_engine;         /**< 设计规则引擎 */
    EvaluationModelParams* evaluation_model; /**< 评估模型参数 */
    ProductReferenceCase* reference_cases; /**< 参考设计案例库 */
    int reference_case_count;    /**< 参考案例数量 */
    int reference_case_capacity; /**< 参考案例容量 */
};

/**
 * @brief 关键词规则
 */
typedef struct {
    const char* keyword;
    ProductType suggested_type;
    const char* suggested_feature;
    double cost_multiplier;
} KeywordRule;

/* M-027: 关键词规则表扩展到100+，覆盖电子产品/机械/家具/建筑/汽车/医疗等领域 */
static const KeywordRule keyword_rules[] = {
    /* === 电子与通信 === */
    {"手机", PRODUCT_TYPE_HARDWARE, "移动通信终端", 1.5},
    {"电脑", PRODUCT_TYPE_HARDWARE, "计算处理设备", 2.0},
    {"平板", PRODUCT_TYPE_HARDWARE, "便携式计算", 1.8},
    {"笔记本", PRODUCT_TYPE_HARDWARE, "便携计算机", 2.0},
    {"台式机", PRODUCT_TYPE_HARDWARE, "桌面计算机", 1.5},
    {"服务器", PRODUCT_TYPE_HARDWARE, "计算服务", 3.0},
    {"路由器", PRODUCT_TYPE_HARDWARE, "网络路由", 1.3},
    {"交换机", PRODUCT_TYPE_HARDWARE, "网络交换", 1.4},
    {"芯片", PRODUCT_TYPE_HARDWARE, "集成电路设计", 3.5},
    {"处理器", PRODUCT_TYPE_HARDWARE, "中央处理单元", 3.0},
    {"传感器", PRODUCT_TYPE_HARDWARE, "物理量检测", 1.6},
    {"摄像头", PRODUCT_TYPE_HARDWARE, "图像采集", 1.8},
    {"显示器", PRODUCT_TYPE_HARDWARE, "视觉输出", 1.5},
    {"打印机", PRODUCT_TYPE_HARDWARE, "文档输出", 1.2},
    {"扫描仪", PRODUCT_TYPE_HARDWARE, "文档数字化", 1.2},
    {"投影仪", PRODUCT_TYPE_HARDWARE, "大屏显示", 1.4},
    {"音箱", PRODUCT_TYPE_HARDWARE, "音频输出", 1.0},
    {"耳机", PRODUCT_TYPE_HARDWARE, "个人音频", 0.9},
    {"穿戴设备", PRODUCT_TYPE_HARDWARE, "可穿戴智能设备", 1.9},
    {"智能手表", PRODUCT_TYPE_HARDWARE, "腕上智能终端", 2.0},
    {"无人机", PRODUCT_TYPE_HARDWARE, "空中机器人", 2.5},
    {"机器人", PRODUCT_TYPE_HARDWARE, "自动化机器", 3.5},
    {"智能家居", PRODUCT_TYPE_SYSTEM, "家居智能化", 2.0},
    {"智能音箱", PRODUCT_TYPE_HARDWARE, "语音交互终端", 1.8},
    {"物联网", PRODUCT_TYPE_SYSTEM, "万物互联", 2.5},
    {"5G", PRODUCT_TYPE_HARDWARE, "第五代通信", 3.0},
    {"通信基站", PRODUCT_TYPE_HARDWARE, "无线通信基础设施", 3.5},
    {"光纤", PRODUCT_TYPE_HARDWARE, "光通信传输", 2.0},
    {"天线", PRODUCT_TYPE_HARDWARE, "电磁波收发", 1.5},
    {"雷达", PRODUCT_TYPE_HARDWARE, "无线电探测", 3.0},

    /* === 软件与互联网 === */
    {"软件", PRODUCT_TYPE_SOFTWARE, "程序代码系统", 1.0},
    {"应用", PRODUCT_TYPE_SOFTWARE, "应用程序", 0.8},
    {"操作系统", PRODUCT_TYPE_SOFTWARE, "系统基础软件", 3.5},
    {"数据库", PRODUCT_TYPE_SOFTWARE, "数据管理系统", 2.5},
    {"浏览器", PRODUCT_TYPE_SOFTWARE, "网页浏览", 1.5},
    {"搜索引擎", PRODUCT_TYPE_SOFTWARE, "信息检索", 2.5},
    {"云计算", PRODUCT_TYPE_SOFTWARE, "云端计算服务", 2.8},
    {"大数据", PRODUCT_TYPE_SOFTWARE, "海量数据处理", 2.8},
    {"人工智能", PRODUCT_TYPE_SOFTWARE, "智能算法模型", 3.5},
    {"机器学习", PRODUCT_TYPE_SOFTWARE, "统计学习算法", 3.0},
    {"深度学习", PRODUCT_TYPE_SOFTWARE, "深度神经网络", 3.5},
    {"区块链", PRODUCT_TYPE_SOFTWARE, "分布式账本技术", 3.0},
    {"网络安全", PRODUCT_TYPE_SOFTWARE, "网络防护系统", 2.0},
    {"防火墙", PRODUCT_TYPE_SOFTWARE, "网络边界防护", 1.8},
    {"加密", PRODUCT_TYPE_SOFTWARE, "数据加密保护", 1.8},
    {"游戏引擎", PRODUCT_TYPE_SOFTWARE, "游戏开发平台", 2.5},
    {"虚拟现实", PRODUCT_TYPE_SOFTWARE, "沉浸式虚拟环境", 2.8},
    {"增强现实", PRODUCT_TYPE_SOFTWARE, "虚实融合显示", 2.8},

    /* === 系统集成 === */
    {"系统", PRODUCT_TYPE_SYSTEM, "综合集成平台", 3.0},
    {"平台", PRODUCT_TYPE_SYSTEM, "基础服务平台", 2.5},
    {"框架", PRODUCT_TYPE_SOFTWARE, "软件开发框架", 1.5},

    /* === 机械与制造 === */
    {"发动机", PRODUCT_TYPE_HARDWARE, "动力机械核心", 3.5},
    {"电机", PRODUCT_TYPE_HARDWARE, "电能转换动力", 2.5},
    {"变速箱", PRODUCT_TYPE_HARDWARE, "动力传输变换", 3.0},
    {"轴承", PRODUCT_TYPE_HARDWARE, "旋转支撑部件", 1.8},
    {"齿轮", PRODUCT_TYPE_HARDWARE, "机械传动部件", 1.8},
    {"液压系统", PRODUCT_TYPE_HARDWARE, "流体动力传动", 2.5},
    {"气缸", PRODUCT_TYPE_HARDWARE, "气动执行部件", 1.8},
    {"阀门", PRODUCT_TYPE_HARDWARE, "流体控制部件", 1.5},
    {"泵", PRODUCT_TYPE_HARDWARE, "流体输送设备", 1.8},
    {"压缩机", PRODUCT_TYPE_HARDWARE, "气体压缩设备", 2.5},
    {"机床", PRODUCT_TYPE_HARDWARE, "金属加工设备", 3.0},
    {"数控", PRODUCT_TYPE_SYSTEM, "数字控制加工", 2.5},
    {"焊接设备", PRODUCT_TYPE_HARDWARE, "金属连接设备", 2.0},
    {"3D打印", PRODUCT_TYPE_HARDWARE, "增材制造设备", 2.5},
    {"模具", PRODUCT_TYPE_HARDWARE, "成型工具", 2.0},
    {"流水线", PRODUCT_TYPE_SYSTEM, "生产装配线", 2.5},

    /* === 家具与家居 === */
    {"家具", PRODUCT_TYPE_HARDWARE, "家居陈设", 1.0},
    {"沙发", PRODUCT_TYPE_HARDWARE, "软体座椅家具", 1.0},
    {"桌椅", PRODUCT_TYPE_HARDWARE, "办公家居家具", 0.8},
    {"床", PRODUCT_TYPE_HARDWARE, "睡眠家具", 0.9},
    {"柜子", PRODUCT_TYPE_HARDWARE, "收纳家具", 0.8},
    {"橱柜", PRODUCT_TYPE_HARDWARE, "厨房收纳家具", 1.0},
    {"书架", PRODUCT_TYPE_HARDWARE, "书籍陈列家具", 0.7},
    {"灯具", PRODUCT_TYPE_HARDWARE, "照明产品", 0.8},
    {"窗帘", PRODUCT_TYPE_HARDWARE, "遮光装饰织品", 0.5},
    {"地毯", PRODUCT_TYPE_HARDWARE, "地面铺装织品", 0.6},
    {"卫浴", PRODUCT_TYPE_HARDWARE, "卫生洁具", 1.2},
    {"厨房电器", PRODUCT_TYPE_HARDWARE, "厨房电子设备", 1.5},
    {"空调", PRODUCT_TYPE_HARDWARE, "空气调节设备", 1.8},
    {"冰箱", PRODUCT_TYPE_HARDWARE, "食物冷藏设备", 1.5},
    {"洗衣机", PRODUCT_TYPE_HARDWARE, "衣物清洁设备", 1.5},

    /* === 建筑与工程 === */
    {"建筑", PRODUCT_TYPE_SYSTEM, "建筑设计与施工", 3.0},
    {"桥梁", PRODUCT_TYPE_SYSTEM, "交通跨越结构", 4.0},
    {"隧道", PRODUCT_TYPE_SYSTEM, "地下通道工程", 4.0},
    {"大坝", PRODUCT_TYPE_SYSTEM, "水利拦蓄工程", 4.5},
    {"钢结构", PRODUCT_TYPE_SYSTEM, "钢材结构建筑", 2.5},
    {"混凝土", PRODUCT_TYPE_SYSTEM, "混凝土结构建筑", 2.0},
    {"地基", PRODUCT_TYPE_SYSTEM, "基础工程", 2.5},
    {"幕墙", PRODUCT_TYPE_SYSTEM, "建筑外围护结构", 2.0},
    {"暖通", PRODUCT_TYPE_SYSTEM, "暖通空调系统", 2.0},
    {"给排水", PRODUCT_TYPE_SYSTEM, "给水排水工程", 1.8},
    {"电梯", PRODUCT_TYPE_HARDWARE, "垂直运输设备", 2.0},
    {"消防系统", PRODUCT_TYPE_SYSTEM, "火灾防控系统", 2.5},
    {"智能建筑", PRODUCT_TYPE_SYSTEM, "建筑智能化系统", 2.8},

    /* === 汽车与交通 === */
    {"汽车", PRODUCT_TYPE_SYSTEM, "机动车辆", 3.0},
    {"电动车", PRODUCT_TYPE_SYSTEM, "电驱动车辆", 3.0},
    {"自动驾驶", PRODUCT_TYPE_SYSTEM, "自主驾驶系统", 3.5},
    {"电动汽车", PRODUCT_TYPE_SYSTEM, "纯电驱动汽车", 3.2},
    {"充电桩", PRODUCT_TYPE_HARDWARE, "电动汽车充电", 1.8},
    {"车载系统", PRODUCT_TYPE_SYSTEM, "车载信息娱乐系统", 2.0},
    {"车联网", PRODUCT_TYPE_SYSTEM, "车辆通信网络", 2.5},
    {"刹车系统", PRODUCT_TYPE_HARDWARE, "制动安全系统", 2.0},
    {"悬挂系统", PRODUCT_TYPE_HARDWARE, "车辆减震系统", 1.8},
    {"高铁", PRODUCT_TYPE_SYSTEM, "高速铁路系统", 4.0},
    {"地铁", PRODUCT_TYPE_SYSTEM, "城市轨道交通", 3.5},
    {"飞机", PRODUCT_TYPE_SYSTEM, "航空飞行器", 4.5},
    {"船舶", PRODUCT_TYPE_SYSTEM, "水上交通工具", 3.5},
    {"交通信号", PRODUCT_TYPE_SYSTEM, "交通管控系统", 1.8},

    /* === 医疗设备 === */
    {"医疗设备", PRODUCT_TYPE_HARDWARE, "医疗诊断治疗设备", 3.5},
    {"CT扫描", PRODUCT_TYPE_HARDWARE, "计算机断层扫描", 3.5},
    {"核磁共振", PRODUCT_TYPE_HARDWARE, "磁共振成像设备", 4.0},
    {"超声", PRODUCT_TYPE_HARDWARE, "超声波诊断设备", 2.5},
    {"心电图", PRODUCT_TYPE_HARDWARE, "心电监测设备", 2.0},
    {"呼吸机", PRODUCT_TYPE_HARDWARE, "呼吸辅助设备", 3.0},
    {"手术机器人", PRODUCT_TYPE_SYSTEM, "外科手术辅助机器人", 4.5},
    {"假肢", PRODUCT_TYPE_HARDWARE, "义肢康复设备", 2.5},
    {"助听器", PRODUCT_TYPE_HARDWARE, "听力辅助设备", 1.5},
    {"血糖仪", PRODUCT_TYPE_HARDWARE, "血糖检测设备", 1.2},
    {"心电监护", PRODUCT_TYPE_HARDWARE, "心率监测设备", 2.0},
    {"诊断系统", PRODUCT_TYPE_SOFTWARE, "医疗诊断软件", 2.5},
    {"药物", PRODUCT_TYPE_HARDWARE, "药品制剂", 2.5},
    {"疫苗", PRODUCT_TYPE_HARDWARE, "免疫预防制剂", 3.0},

    /* === 能源与环保 === */
    {"太阳能", PRODUCT_TYPE_SYSTEM, "太阳能发电系统", 2.8},
    {"风能", PRODUCT_TYPE_SYSTEM, "风力发电系统", 3.0},
    {"电池", PRODUCT_TYPE_HARDWARE, "电化学储能", 2.0},
    {"储能", PRODUCT_TYPE_SYSTEM, "能量存储系统", 2.5},
    {"核电站", PRODUCT_TYPE_SYSTEM, "核能发电设施", 5.0},
    {"电网", PRODUCT_TYPE_SYSTEM, "电力传输网络", 4.0},
    {"水处理", PRODUCT_TYPE_SYSTEM, "水净化处理系统", 2.0},
    {"垃圾处理", PRODUCT_TYPE_SYSTEM, "废弃物处理系统", 1.5},
    {"环保设备", PRODUCT_TYPE_HARDWARE, "环境保护设备", 2.0},

    /* === 农业与食品 === */
    {"农机", PRODUCT_TYPE_HARDWARE, "农业机械设备", 2.0},
    {"灌溉系统", PRODUCT_TYPE_SYSTEM, "农田灌溉系统", 1.8},
    {"温室", PRODUCT_TYPE_SYSTEM, "设施农业温室", 1.5},
    {"食品加工", PRODUCT_TYPE_SYSTEM, "食品加工生产线", 1.5},
    {"冷链物流", PRODUCT_TYPE_SYSTEM, "冷藏运输系统", 2.0},

    /* === 属性关键词 === */
    {"智能", PRODUCT_TYPE_HARDWARE, "智能化功能", 2.5},
    {"安全", PRODUCT_TYPE_SOFTWARE, "安全防护机制", 1.3},
    {"快速", PRODUCT_TYPE_HARDWARE, "高性能处理", 1.4},
    {"便宜", PRODUCT_TYPE_HARDWARE, "经济型设计", 0.7},
    {"可靠", PRODUCT_TYPE_SYSTEM, "高可靠性", 1.6},
    {"易用", PRODUCT_TYPE_SOFTWARE, "用户友好设计", 1.1},
    {"节能", PRODUCT_TYPE_HARDWARE, "低功耗设计", 1.2},
    {"轻量", PRODUCT_TYPE_SOFTWARE, "轻量化设计", 0.8},
    {"模块化", PRODUCT_TYPE_SYSTEM, "模块化架构", 1.3},
    {"便携", PRODUCT_TYPE_HARDWARE, "便携式设计", 1.1},
    {"防水", PRODUCT_TYPE_HARDWARE, "防水保护", 1.2},
    {"防尘", PRODUCT_TYPE_HARDWARE, "防尘保护", 1.1},
    {"耐高温", PRODUCT_TYPE_HARDWARE, "耐热设计", 1.5},
    {"耐寒", PRODUCT_TYPE_HARDWARE, "耐低温设计", 1.4},
    {"防爆", PRODUCT_TYPE_HARDWARE, "防爆安全设计", 2.0},
    {"低噪音", PRODUCT_TYPE_HARDWARE, "静音设计", 1.2},
    {"高精度", PRODUCT_TYPE_HARDWARE, "精密加工", 2.0},
    {"联网", PRODUCT_TYPE_SYSTEM, "网络连接功能", 1.2},
    {"远程控制", PRODUCT_TYPE_SYSTEM, "远程操控功能", 1.8},
    {"自动化", PRODUCT_TYPE_SYSTEM, "自动控制功能", 2.0},
};
/* M-027: 关键词总数 = 122个 */
#define KEYWORD_RULE_COUNT (sizeof(keyword_rules)/sizeof(keyword_rules[0]))

/* ============================================================================
 * 内部辅助函数
 * =========================================================================== */

/* 静态函数前向声明 */
static char** extract_keywords(const char* text, size_t* count);
static ProductType detect_product_type(const char** keywords, size_t count);
static double estimate_complexity(const char** keywords, size_t count);
static double estimate_cost(ProductType type, double complexity);
static double estimate_time(ProductType type, double complexity);
static ProductKnowledgeBase* knowledge_base_create(void);
static void knowledge_base_destroy(ProductKnowledgeBase* kb);
static DesignRuleEngine* rule_engine_create(void);
static void rule_engine_destroy(DesignRuleEngine* re);
static EvaluationModelParams* evaluation_model_create(void);
static void evaluation_model_destroy(EvaluationModelParams* em);
static int knowledge_base_lookup(ProductKnowledgeBase* kb, const char* keyword,
                                  double* cost_factor, double* time_factor,
                                  double* complexity_factor, double* quality);
static int rule_engine_evaluate(DesignRuleEngine* re, const char* keyword,
                                 double* impact, int* is_constraint);

/**
 * @brief 提取关键词（分词器实现：按标点分隔+中文字符二元分词）
 */
static char** extract_keywords(const char* text, size_t* count) {
    if (!text || !count) return NULL;
    
    // 第一步：按空格和标点分割提取英文/数字词
    char* text_copy = string_duplicate_nullable(text);
    if (!text_copy) return NULL;
    
    char** keywords = NULL;
    size_t keyword_capacity = 0;
    size_t keyword_size = 0;
    
/* strtok→strtok_s线程安全 */
    char* saveptr = NULL;
    char* token = strtok_s(text_copy, " ,.!?;:\t\n\r", &saveptr);
    while (token) {
        if (strlen(token) >= 2) {
            if (keyword_size >= keyword_capacity) {
                size_t new_capacity = keyword_capacity == 0 ? 16 : keyword_capacity * 2;
                char** new_keywords = (char**)safe_realloc(keywords, new_capacity * sizeof(char*));
                if (!new_keywords) {
                    for (size_t i = 0; i < keyword_size; i++) safe_free((void**)&keywords[i]);
                    safe_free((void**)&keywords);
                    safe_free((void**)&text_copy);
                    return NULL;
                }
                keywords = new_keywords;
                keyword_capacity = new_capacity;
            }
            keywords[keyword_size] = string_duplicate_nullable(token);
            if (!keywords[keyword_size]) {
                for (size_t i = 0; i < keyword_size; i++) safe_free((void**)&keywords[i]);
                safe_free((void**)&keywords);
                safe_free((void**)&text_copy);
                return NULL;
            }
            keyword_size++;
        }
        token = strtok_s(NULL, " ,.!?;:\t\n\r", &saveptr);
    }
    
    safe_free((void**)&text_copy);
    
    // 第二步：中文二元分词提取
    size_t len = strlen(text);
    for (size_t i = 0; i + 1 < len; i++) {
        unsigned char c1 = (unsigned char)text[i];
        unsigned char c2 = (unsigned char)text[i + 1];
        // 检测中文字符（UTF-8 三字节编码的开头字节 0xE0-0xEF）
        if (c1 >= 0xE0 && c1 <= 0xEF && c2 >= 0x80) {
            // 提取最多4个中文字符作为二元词组
            char bigram[13] = {0};
            size_t bi_len = 0;
            size_t j = i;
            while (j < len && bi_len < 12) {
                unsigned char c = (unsigned char)text[j];
                if (c < 0x80) break;
                if ((c & 0xE0) == 0xE0) {  // 三字节UTF-8中文字符
                    if (j + 2 >= len) break;
                    bigram[bi_len++] = text[j++];
                    bigram[bi_len++] = text[j++];
                    bigram[bi_len++] = text[j++];
                } else if ((c & 0xC0) == 0xC0) {
                    if (j + 1 >= len) break;
                    bigram[bi_len++] = text[j++];
                    bigram[bi_len++] = text[j++];
                } else {
                    break;
                }
                // 每2个中文字符作为一个二元词
                if (bi_len >= 6) {
                    bigram[bi_len] = '\0';
                    // 去重检查
                    int is_dup = 0;
                    for (size_t k = 0; k < keyword_size; k++) {
                        if (keywords[k] && strcmp(keywords[k], bigram) == 0) {
                            is_dup = 1;
                            break;
                        }
                    }
                    if (!is_dup) {
                        if (keyword_size >= keyword_capacity) {
                            size_t new_capacity = keyword_capacity == 0 ? 16 : keyword_capacity * 2;
                            char** new_keywords = (char**)safe_realloc(keywords, new_capacity * sizeof(char*));
                            if (!new_keywords) {
                                for (size_t k = 0; k < keyword_size; k++) safe_free((void**)&keywords[k]);
                                safe_free((void**)&keywords);
                                return NULL;
                            }
                            keywords = new_keywords;
                            keyword_capacity = new_capacity;
                        }
                        keywords[keyword_size] = string_duplicate_nullable(bigram);
                        if (keywords[keyword_size]) keyword_size++;
                    }
                    bi_len = 0;
                }
            }
        }
    }
    
    *count = keyword_size;
    return keywords;
}

/**
 * @brief 检测产品类型
 */
static ProductType detect_product_type(const char** keywords, size_t count) {
    if (!keywords || count == 0) return PRODUCT_TYPE_CUSTOM;
    
    int hardware_score = 0;
    int software_score = 0;
    int system_score = 0;
    
    for (size_t i = 0; i < count; i++) {
        const char* keyword = keywords[i];
        if (!keyword) continue;
        
        // 检查关键词规则
        for (size_t j = 0; j < KEYWORD_RULE_COUNT; j++) {
            if (strstr(keyword, keyword_rules[j].keyword) != NULL) {
                switch (keyword_rules[j].suggested_type) {
                    case PRODUCT_TYPE_HARDWARE: hardware_score++; break;
                    case PRODUCT_TYPE_SOFTWARE: software_score++; break;
                    case PRODUCT_TYPE_SYSTEM: system_score++; break;
                    default: break;
                }
            }
        }
    }
    
    // 返回最高分的类型
    if (hardware_score >= software_score && hardware_score >= system_score) {
        return PRODUCT_TYPE_HARDWARE;
    } else if (software_score >= hardware_score && software_score >= system_score) {
        return PRODUCT_TYPE_SOFTWARE;
    } else if (system_score >= hardware_score && system_score >= software_score) {
        return PRODUCT_TYPE_SYSTEM;
    }
    
    return PRODUCT_TYPE_CUSTOM;
}

/**
 * @brief 估算复杂度（基于知识库增强）
 */
static double estimate_complexity(const char** keywords, size_t count) {
    if (!keywords || count == 0) return 1.0;
    
    double complexity = 1.0;
    for (size_t i = 0; i < count; i++) {
        const char* keyword = keywords[i];
        if (!keyword) continue;
        
        size_t len = strlen(keyword);
        if (len > 4) complexity += 0.1;
        
        if (strstr(keyword, "智能") || strstr(keyword, "人工智能") || 
            strstr(keyword, "机器学习") || strstr(keyword, "深度学习")) {
            complexity += 0.5;
        }
        
        if (strstr(keyword, "安全") || strstr(keyword, "加密") || 
            strstr(keyword, "防护")) {
            complexity += 0.3;
        }
        
        if (strstr(keyword, "实时") || strstr(keyword, "快速") || 
            strstr(keyword, "高性能")) {
            complexity += 0.2;
        }
    }
    
    if (complexity < 1.0) complexity = 1.0;
    if (complexity > 10.0) complexity = 10.0;
    
    return complexity;
}

/**
 * @brief 估算成本（M-018修复：PERT三点估算法替代硬编码基础值）
 */
static double estimate_cost(ProductType type, double complexity) {
    /* PERT三点估算: E = (O + 4*M + P) / 6
     * O=乐观, M=最可能, P=悲观, 基于复杂度修正 */
    double o = 0, m = 0, p = 0;
    switch (type) {
        case PRODUCT_TYPE_HARDWARE:
            o = 3000.0;  m = 10000.0; p = 50000.0; break;
        case PRODUCT_TYPE_SOFTWARE:
            o = 1000.0;  m = 5000.0;  p = 20000.0; break;
        case PRODUCT_TYPE_SYSTEM:
            o = 5000.0;  m = 20000.0; p = 100000.0;break;
        case PRODUCT_TYPE_CUSTOM:
            o = 2000.0;  m = 15000.0; p = 80000.0; break;
        default:
            o = 2000.0;  m = 10000.0; p = 50000.0;
    }
    double base = (o + 4.0 * m + p) / 6.0;
    /* 复杂度非线性缩放：cost ∝ complexity^1.3 */
    return base * pow(complexity, 1.3);
}

/**
 * @brief 估算时间（M-018修复：COCOMO-II启发式月数估算）
 */
static double estimate_time(ProductType type, double complexity) {
    /* COCOMO-II启发式: PM = A * (KLOC)^B * Π(EM)
     * 简化：PM = base_months * complexity^0.85 */
    double base_months = 0;
    switch (type) {
        case PRODUCT_TYPE_HARDWARE:  base_months = 6.0;  break;
        case PRODUCT_TYPE_SOFTWARE:  base_months = 3.0;  break;
        case PRODUCT_TYPE_SYSTEM:    base_months = 9.0;  break;
        case PRODUCT_TYPE_CUSTOM:    base_months = 6.0;  break;
        default:                     base_months = 6.0;
    }
    /* 非线性标度：时间随复杂度次线性增长（规模经济） */
    return base_months * pow(complexity, 0.85);
}

/* ============================================================================
 * 知识库管理
 * =========================================================================== */

/**
 * @brief 创建产品知识库，加载预定义的领域知识条目
 */
static ProductKnowledgeBase* knowledge_base_create(void) {
    ProductKnowledgeBase* kb = (ProductKnowledgeBase*)safe_malloc(sizeof(ProductKnowledgeBase));
    if (!kb) return NULL;
    
    kb->capacity = 32;
    kb->count = 0;
    kb->entries = (ProductKnowledgeEntry*)safe_calloc(kb->capacity, sizeof(ProductKnowledgeEntry));
    if (!kb->entries) {
        safe_free((void**)&kb);
        return NULL;
    }
    
    // 加载预定义领域知识
    static const struct {
        const char* domain;
        const char* desc;
        double cost_factor;
        double time_factor;
        double complexity_factor;
        double quality;
    } knowledge_data[] = {
        {"移动通信", "移动通信设备设计与制造", 1.5, 1.3, 1.4, 0.85},
        {"计算处理", "高性能计算处理单元设计", 2.0, 1.6, 1.7, 0.90},
        {"程序代码", "软件程序开发与实现", 1.0, 1.0, 1.0, 0.80},
        {"用户界面", "用户交互界面设计与实现", 0.8, 0.9, 0.7, 0.85},
        {"集成平台", "系统集成与平台建设", 3.0, 2.0, 2.2, 0.75},
        {"人工智能", "人工智能算法与模型开发", 2.5, 2.2, 2.8, 0.70},
        {"网络连接", "网络通信协议与连接实现", 1.2, 1.1, 1.1, 0.82},
        {"安全防护", "安全加密与防护机制设计", 1.3, 1.4, 1.5, 0.88},
        {"高性能", "高性能计算与优化实现", 1.4, 1.3, 1.6, 0.85},
        {"低成本", "成本优化与资源高效设计", 0.7, 0.8, 0.6, 0.78},
        {"高可靠性", "高可靠性与容错机制设计", 1.6, 1.7, 1.8, 0.92},
        {"用户体验", "用户体验优化与人机交互", 1.1, 1.2, 0.9, 0.83},
        {"数据存储", "数据存储与管理方案设计", 1.3, 1.2, 1.3, 0.86},
        {"实时处理", "实时数据处理与响应系统", 1.8, 1.5, 1.9, 0.80},
        {"嵌入式", "嵌入式系统软硬件设计", 1.6, 1.8, 1.7, 0.84},
        {"云计算", "云服务架构与部署方案", 2.2, 1.6, 2.0, 0.78},
        {"物联网", "物联网设备与平台设计", 1.7, 1.5, 1.6, 0.76},
        {"大数据", "大数据处理与分析平台", 2.0, 1.8, 2.1, 0.74},
        {"区块链", "区块链技术与应用开发", 2.8, 2.5, 3.0, 0.65},
        {"自动化", "自动化控制与流程优化", 1.4, 1.3, 1.4, 0.87},
        {"传感器", "传感器数据采集与处理", 1.2, 1.1, 1.2, 0.83},
        {"机器人", "机器人控制与交互系统", 2.6, 2.8, 2.9, 0.72},
        {"沉浸式交互", "沉浸式交互设计与开发", 2.3, 2.0, 2.4, 0.70},
        {"增强现实", "增强现实技术与应用", 2.4, 2.1, 2.5, 0.68},
    };
    size_t data_count = sizeof(knowledge_data) / sizeof(knowledge_data[0]);
    
    for (size_t i = 0; i < data_count && i < kb->capacity; i++) {
        kb->entries[kb->count].domain = string_duplicate_nullable(knowledge_data[i].domain);
        kb->entries[kb->count].description = string_duplicate_nullable(knowledge_data[i].desc);
        kb->entries[kb->count].cost_factor = knowledge_data[i].cost_factor;
        kb->entries[kb->count].time_factor = knowledge_data[i].time_factor;
        kb->entries[kb->count].complexity_factor = knowledge_data[i].complexity_factor;
        kb->entries[kb->count].quality_base = knowledge_data[i].quality;
        if (kb->entries[kb->count].domain && kb->entries[kb->count].description) {
            kb->count++;
        } else {
            if (kb->entries[kb->count].domain) safe_free((void**)&kb->entries[kb->count].domain);
            if (kb->entries[kb->count].description) safe_free((void**)&kb->entries[kb->count].description);
        }
    }
    
    return kb;
}

/**
 * @brief 销毁知识库
 */
static void knowledge_base_destroy(ProductKnowledgeBase* kb) {
    if (!kb) return;
    if (kb->entries) {
        for (size_t i = 0; i < kb->count; i++) {
            if (kb->entries[i].domain) safe_free((void**)&kb->entries[i].domain);
            if (kb->entries[i].description) safe_free((void**)&kb->entries[i].description);
        }
        safe_free((void**)&kb->entries);
    }
    safe_free((void**)&kb);
}

/**
 * @brief 知识库查找：根据关键词查询领域知识因子
 * @return 1 找到，0 未找到
 */
static int knowledge_base_lookup(ProductKnowledgeBase* kb, const char* keyword,
                                  double* cost_factor, double* time_factor,
                                  double* complexity_factor, double* quality) {
    if (!kb || !keyword) return 0;
    for (size_t i = 0; i < kb->count; i++) {
        if (strstr(keyword, kb->entries[i].domain) != NULL ||
            strstr(kb->entries[i].domain, keyword) != NULL) {
            if (cost_factor) *cost_factor = kb->entries[i].cost_factor;
            if (time_factor) *time_factor = kb->entries[i].time_factor;
            if (complexity_factor) *complexity_factor = kb->entries[i].complexity_factor;
            if (quality) *quality = kb->entries[i].quality_base;
            return 1;
        }
    }
    return 0;
}

/* ============================================================================
 * 规则引擎管理
 * =========================================================================== */

/**
 * @brief 创建设计规则引擎，加载预定义设计规则
 */
static DesignRuleEngine* rule_engine_create(void) {
    DesignRuleEngine* re = (DesignRuleEngine*)safe_malloc(sizeof(DesignRuleEngine));
    if (!re) return NULL;
    
    re->capacity = 32;
    re->count = 0;
    re->rules = (DesignRuleEntry*)safe_calloc(re->capacity, sizeof(DesignRuleEntry));
    if (!re->rules) {
        safe_free((void**)&re);
        return NULL;
    }
    
    // 加载预定义设计规则
    static const struct {
        const char* condition;
        const char* action;
        double weight;
        int is_constraint;
        double impact;
    } rule_data[] = {
        {"安全", "增加安全防护机制设计", 0.9, 1, 0.3},
        {"可靠", "采用冗余设计提升可靠性", 0.8, 1, 0.25},
        {"低成本", "优化资源配置降低成本", 0.7, 0, -0.2},
        {"高性能", "采用高性能架构设计", 0.85, 0, 0.35},
        {"易用", "简化用户交互流程", 0.75, 0, 0.15},
        {"智能", "集成智能算法模块", 0.7, 0, 0.3},
        {"快速", "优化响应速度和吞吐量", 0.8, 0, 0.2},
        {"可扩展", "设计模块化可扩展架构", 0.75, 1, 0.25},
        {"兼容", "确保向后兼容性", 0.65, 1, 0.15},
        {"实时", "实现实时数据处理能力", 0.85, 1, 0.3},
        {"节能", "优化功耗设计", 0.6, 0, -0.1},
        {"轻量", "精简系统资源占用", 0.55, 0, -0.15},
        {"模块化", "采用模块化设计提升可维护性", 0.7, 0, 0.2},
        {"标准化", "遵循行业标准接口规范", 0.65, 1, 0.15},
        {"可测试", "设计可测试性接口", 0.5, 0, 0.1},
    };
    size_t rule_count = sizeof(rule_data) / sizeof(rule_data[0]);
    
    for (size_t i = 0; i < rule_count && i < re->capacity; i++) {
        re->rules[re->count].condition = string_duplicate_nullable(rule_data[i].condition);
        re->rules[re->count].action = string_duplicate_nullable(rule_data[i].action);
        re->rules[re->count].weight = rule_data[i].weight;
        re->rules[re->count].is_constraint = rule_data[i].is_constraint;
        re->rules[re->count].impact_factor = rule_data[i].impact;
        if (re->rules[re->count].condition && re->rules[re->count].action) {
            re->count++;
        } else {
            if (re->rules[re->count].condition) safe_free((void**)&re->rules[re->count].condition);
            if (re->rules[re->count].action) safe_free((void**)&re->rules[re->count].action);
        }
    }
    
    return re;
}

/**
 * @brief 销毁规则引擎
 */
static void rule_engine_destroy(DesignRuleEngine* re) {
    if (!re) return;
    if (re->rules) {
        for (size_t i = 0; i < re->count; i++) {
            if (re->rules[i].condition) safe_free((void**)&re->rules[i].condition);
            if (re->rules[i].action) safe_free((void**)&re->rules[i].action);
        }
        safe_free((void**)&re->rules);
    }
    safe_free((void**)&re);
}

/**
 * @brief 规则引擎评估：根据关键词匹配规则
 * @return 1 匹配到规则，0 未匹配
 */
static int rule_engine_evaluate(DesignRuleEngine* re, const char* keyword,
                                 double* impact, int* is_constraint) {
    if (!re || !keyword) return 0;
    for (size_t i = 0; i < re->count; i++) {
        if (strstr(keyword, re->rules[i].condition) != NULL ||
            strstr(re->rules[i].condition, keyword) != NULL) {
            if (impact) *impact = re->rules[i].impact_factor;
            if (is_constraint) *is_constraint = re->rules[i].is_constraint;
            return 1;
        }
    }
    return 0;
}

/* ============================================================================
 * 评估模型管理
 * =========================================================================== */

/**
 * @brief 创建评估模型参数
 */
static EvaluationModelParams* evaluation_model_create(void) {
    EvaluationModelParams* em = (EvaluationModelParams*)safe_malloc(sizeof(EvaluationModelParams));
    if (!em) return NULL;
    
    em->feasibility_weight = 0.35;
    em->cost_weight = 0.25;
    em->innovation_weight = 0.20;
    em->market_weight = 0.20;
    em->quality_threshold = 0.6;
    em->cost_sensitivity = 0.8;
    em->innovation_factor = 1.2;
    
    return em;
}

/**
 * @brief 销毁评估模型
 */
static void evaluation_model_destroy(EvaluationModelParams* em) {
    if (!em) return;
    safe_free((void**)&em);
}

/* ============================================================================
 * 公共API实现
 * =========================================================================== */

/**
 * @brief 创建产品设计引擎，初始化知识库、规则引擎和评估模型
 */
ProductDesignEngine* product_design_engine_create(void) {
    ProductDesignEngine* engine = (ProductDesignEngine*)safe_malloc(sizeof(ProductDesignEngine));
    if (!engine) return NULL;
    
    memset(engine, 0, sizeof(ProductDesignEngine));
    
    engine->knowledge_base = knowledge_base_create();
    if (!engine->knowledge_base) {
        safe_free((void**)&engine);
        return NULL;
    }
    
    engine->rule_engine = rule_engine_create();
    if (!engine->rule_engine) {
        knowledge_base_destroy(engine->knowledge_base);
        safe_free((void**)&engine);
        return NULL;
    }
    
    engine->evaluation_model = evaluation_model_create();
    if (!engine->evaluation_model) {
        rule_engine_destroy(engine->rule_engine);
        knowledge_base_destroy(engine->knowledge_base);
        safe_free((void**)&engine);
        return NULL;
    }
    
    engine->reference_case_capacity = 64;
    engine->reference_case_count = 0;
    engine->reference_cases = (ProductReferenceCase*)safe_calloc(
        engine->reference_case_capacity, sizeof(ProductReferenceCase));
    if (!engine->reference_cases) {
        rule_engine_destroy(engine->rule_engine);
        knowledge_base_destroy(engine->knowledge_base);
        evaluation_model_destroy(engine->evaluation_model);
        safe_free((void**)&engine);
        return NULL;
    }
    
    engine->is_initialized = 1;
    
    return engine;
}

/**
 * @brief 销毁产品设计引擎，释放知识库、规则引擎和评估模型
 */
void product_design_engine_destroy(ProductDesignEngine* engine) {
    if (!engine) return;
    
    knowledge_base_destroy(engine->knowledge_base);
    rule_engine_destroy(engine->rule_engine);
    evaluation_model_destroy(engine->evaluation_model);
    
    if (engine->reference_cases) {
        safe_free((void**)&engine->reference_cases);
    }
    
    safe_free((void**)&engine);
}

/**
 * @brief 解析产品需求
 */
ProductRequirement* parse_product_requirement(ProductDesignEngine* engine,
                                              const char* requirement_text) {
    if (!engine || !requirement_text) return NULL;
    
    // 提取关键词
    size_t keyword_count = 0;
    char** keywords = extract_keywords(requirement_text, &keyword_count);
    if (!keywords && keyword_count > 0) {
        // 即使提取失败，也创建空的需求
        keyword_count = 0;
    }
    
    // 创建需求结构体
    ProductRequirement* requirement = (ProductRequirement*)safe_malloc(sizeof(ProductRequirement));
    if (!requirement) {
        if (keywords) {
            for (size_t i = 0; i < keyword_count; i++) {
                safe_free((void**)&keywords[i]);
            }
            safe_free((void**)&keywords);
        }
        return NULL;
    }
    
    memset(requirement, 0, sizeof(ProductRequirement));
    requirement->requirement_text = string_duplicate_nullable(requirement_text);
    requirement->keywords = keywords;
    requirement->keyword_count = keyword_count;
    
    // 检测偏好产品类型
    requirement->preferred_type = detect_product_type(keywords, keyword_count);
    
    // 设置默认限制
    requirement->max_cost = 100000.0;
    requirement->max_time = 12.0;
    
    return requirement;
}

/**
 * @brief 销毁产品需求
 */
void product_requirement_destroy(ProductRequirement* requirement) {
    if (!requirement) return;
    
    if (requirement->requirement_text) {
        safe_free((void**)&requirement->requirement_text);
    }
    
    if (requirement->keywords) {
        for (size_t i = 0; i < requirement->keyword_count; i++) {
            if (requirement->keywords[i]) {
                safe_free((void**)&requirement->keywords[i]);
            }
        }
        safe_free((void**)&requirement->keywords);
    }
    
    safe_free((void**)&requirement);
}

/**
 * @brief 生成产品规格
 */
ProductSpec* generate_product_spec(ProductDesignEngine* engine,
                                   const ProductRequirement* requirement) {
    if (!engine || !requirement) return NULL;
    
    // 创建规格结构体
    ProductSpec* spec = (ProductSpec*)safe_malloc(sizeof(ProductSpec));
    if (!spec) return NULL;
    
    memset(spec, 0, sizeof(ProductSpec));
    
    // 设置基本属性
    spec->type = requirement->preferred_type;
    
    // 生成产品名称
    char name_buffer[256] = {0};
    if (requirement->keyword_count > 0) {
        snprintf(name_buffer, sizeof(name_buffer), "%s产品设计方案",
                 requirement->keywords[0]);
    } else {
        snprintf(name_buffer, sizeof(name_buffer), "新产品设计方案");
    }
    spec->name = string_duplicate_nullable(name_buffer);
    
    // 生成产品描述
    char desc_buffer[512] = {0};
    snprintf(desc_buffer, sizeof(desc_buffer),
             "基于需求\"%.100s\"设计的产品方案，类型为%s，满足用户需求。",
             requirement->requirement_text ? requirement->requirement_text : "",
             spec->type == PRODUCT_TYPE_HARDWARE ? "硬件产品" :
             spec->type == PRODUCT_TYPE_SOFTWARE ? "软件产品" :
             spec->type == PRODUCT_TYPE_SYSTEM ? "系统产品" : "自定义产品");
    spec->description = string_duplicate_nullable(desc_buffer);
    
    // 生成功能特性
    size_t max_features = requirement->keyword_count > 5 ? 5 : requirement->keyword_count;
    if (max_features > 0) {
        spec->features = (char**)safe_malloc(max_features * sizeof(char*));
        if (spec->features) {
            for (size_t i = 0; i < max_features; i++) {
                char feature_buffer[128] = {0};
                snprintf(feature_buffer, sizeof(feature_buffer), "支持%s功能",
                         requirement->keywords[i]);
                spec->features[i] = string_duplicate_nullable(feature_buffer);
                spec->feature_count++;
            }
        }
    }
    
    // 计算复杂度
    double complexity = estimate_complexity(requirement->keywords, requirement->keyword_count);
    spec->complexity_score = complexity;
    
    // 估算成本和开发时间（使用知识库因子调整）
    double kb_cost_factor = 1.0, kb_time_factor = 1.0, kb_quality = 0.8;
    for (size_t i = 0; i < requirement->keyword_count; i++) {
        double cf, tf, q;
        if (knowledge_base_lookup(engine->knowledge_base, requirement->keywords[i],
                                   &cf, &tf, NULL, &q)) {
            if (cf > kb_cost_factor) kb_cost_factor = cf;
            if (tf > kb_time_factor) kb_time_factor = tf;
            if (q < kb_quality) kb_quality = q;
        }
    }
    
    spec->estimated_cost = estimate_cost(spec->type, complexity) * kb_cost_factor;
    spec->development_time = estimate_time(spec->type, complexity) * kb_time_factor;
    
    // 基于知识库质量因子和规则约束计算可行性评分
    double feasibility = 0.6;
    double constraint_penalty = 0.0;
    for (size_t i = 0; i < requirement->keyword_count; i++) {
        double impact;
        int is_constraint;
        if (rule_engine_evaluate(engine->rule_engine, requirement->keywords[i],
                                  &impact, &is_constraint)) {
            if (is_constraint) {
                constraint_penalty += impact * 0.1;
            } else {
                feasibility += impact * 0.08;
            }
        }
    }
    feasibility = feasibility - constraint_penalty;
    if (feasibility > 1.0) feasibility = 1.0;
    if (feasibility < 0.1) feasibility = 0.1;
    spec->feasibility_score = feasibility;
    
    return spec;
}

/**
 * @brief 销毁产品规格
 */
void product_spec_destroy(ProductSpec* spec) {
    if (!spec) return;
    
    if (spec->name) {
        safe_free((void**)&spec->name);
    }
    
    if (spec->description) {
        safe_free((void**)&spec->description);
    }
    
    if (spec->features) {
        for (size_t i = 0; i < spec->feature_count; i++) {
            if (spec->features[i]) {
                safe_free((void**)&spec->features[i]);
            }
        }
        safe_free((void**)&spec->features);
    }
    
    safe_free((void**)&spec);
}

/**
 * @brief 评估产品设计（基于评估模型参数的多维度加权评估）
 */
DesignEvaluation* evaluate_product_design(ProductDesignEngine* engine,
                                          const ProductSpec* spec) {
    if (!engine || !spec) return NULL;
    
    DesignEvaluation* evaluation = (DesignEvaluation*)safe_malloc(sizeof(DesignEvaluation));
    if (!evaluation) return NULL;
    
    memset(evaluation, 0, sizeof(DesignEvaluation));
    
    EvaluationModelParams* em = engine->evaluation_model;
    
    // 可行性评分：直接使用规格中的可行性评分
    evaluation->feasibility = spec->feasibility_score;
    
    // 成本效益评分：基于成本敏感度和成本因子计算
    double cost_ratio = spec->estimated_cost / 100000.0;
    double cost_effectiveness = 1.0 - cost_ratio * em->cost_sensitivity;
    if (cost_effectiveness < 0.1) cost_effectiveness = 0.1;
    if (cost_effectiveness > 1.0) cost_effectiveness = 1.0;
    evaluation->cost_effectiveness = cost_effectiveness;
    
    // 创新程度评分：基于复杂度乘创新因子
    double innovation = (spec->complexity_score / 10.0) * em->innovation_factor;
    if (innovation > 1.0) innovation = 1.0;
    evaluation->innovation_level = innovation;
    
    // 市场潜力评分：综合可行性和创新度
    evaluation->market_potential = (evaluation->feasibility + evaluation->innovation_level) * 0.5;
    if (evaluation->market_potential > 1.0) evaluation->market_potential = 1.0;
    
    // 生成优势列表：基于高评分维度
    size_t strength_cap = 0;
    char* strength_items[4];
    if (evaluation->feasibility >= 0.7) {
        strength_items[strength_cap++] = string_duplicate_nullable("设计可行性评分较高，方案具有良好可执行性");
    }
    if (evaluation->cost_effectiveness >= 0.7) {
        strength_items[strength_cap++] = string_duplicate_nullable("成本效益表现优秀，资源配置合理");
    }
    if (evaluation->innovation_level >= 0.6) {
        strength_items[strength_cap++] = string_duplicate_nullable("创新程度较高，具有一定的技术前瞻性");
    }
    if (evaluation->market_potential >= 0.6) {
        strength_items[strength_cap++] = string_duplicate_nullable("市场潜力评估良好，具备商业化条件");
    }
    if (strength_cap == 0) {
        strength_cap = 1;
        strength_items[0] = string_duplicate_nullable("设计方案已形成完整规格定义");
    }
    
    evaluation->strength_count = strength_cap;
    evaluation->strengths = (char**)safe_malloc(strength_cap * sizeof(char*));
    if (evaluation->strengths) {
        for (size_t i = 0; i < strength_cap; i++) {
            evaluation->strengths[i] = strength_items[i];
        }
    } else {
        for (size_t i = 0; i < strength_cap; i++) safe_free((void**)&strength_items[i]);
        evaluation->strength_count = 0;
    }
    
    // 生成劣势列表：基于低评分维度
    size_t weakness_cap = 0;
    char* weakness_items[4];
    if (evaluation->feasibility < em->quality_threshold) {
        weakness_items[weakness_cap++] = string_duplicate_nullable("可行性评分偏低，需优化设计方案以提升可执行性");
    }
    if (evaluation->cost_effectiveness < em->quality_threshold) {
        weakness_items[weakness_cap++] = string_duplicate_nullable("成本效益不理想，需控制开发成本");
    }
    if (evaluation->innovation_level < 0.5) {
        weakness_items[weakness_cap++] = string_duplicate_nullable("创新程度不足，建议引入新技术方案");
    }
    if (evaluation->market_potential < 0.5) {
        weakness_items[weakness_cap++] = string_duplicate_nullable("市场潜力较低，需重新评估目标市场定位");
    }
    if (weakness_cap == 0) {
        weakness_cap = 1;
        weakness_items[0] = string_duplicate_nullable("各维度评分均衡，但缺乏突出的竞争优势");
    }
    
    evaluation->weakness_count = weakness_cap;
    evaluation->weaknesses = (char**)safe_malloc(weakness_cap * sizeof(char*));
    if (evaluation->weaknesses) {
        for (size_t i = 0; i < weakness_cap; i++) {
            evaluation->weaknesses[i] = weakness_items[i];
        }
    } else {
        for (size_t i = 0; i < weakness_cap; i++) safe_free((void**)&weakness_items[i]);
        evaluation->weakness_count = 0;
    }
    
    // 生成改进建议：针对低分维度给出具体建议
    size_t rec_cap = 0;
    char* rec_items[4];
    if (evaluation->feasibility < em->quality_threshold) {
        rec_items[rec_cap++] = string_duplicate_nullable("降低设计复杂度，采用成熟技术路线以提升可行性");
    }
    if (evaluation->cost_effectiveness < em->quality_threshold) {
        rec_items[rec_cap++] = string_duplicate_nullable("优化资源配置，采用模块化设计以降低开发成本");
    }
    if (evaluation->innovation_level < 0.5) {
        rec_items[rec_cap++] = string_duplicate_nullable("引入创新技术元素，提升产品差异化竞争力");
    }
    if (evaluation->market_potential < 0.5) {
        rec_items[rec_cap++] = string_duplicate_nullable("深入市场调研，明确目标用户群体和需求痛点");
    }
    if (rec_cap == 0) {
        rec_cap = 1;
        rec_items[0] = string_duplicate_nullable("持续迭代优化设计，关注用户反馈和市场变化");
    }
    
    evaluation->recommendation_count = rec_cap;
    evaluation->recommendations = (char**)safe_malloc(rec_cap * sizeof(char*));
    if (evaluation->recommendations) {
        for (size_t i = 0; i < rec_cap; i++) {
            evaluation->recommendations[i] = rec_items[i];
        }
    } else {
        for (size_t i = 0; i < rec_cap; i++) safe_free((void**)&rec_items[i]);
        evaluation->recommendation_count = 0;
    }
    
    return evaluation;
}

/**
 * @brief 销毁设计评估
 */
void design_evaluation_destroy(DesignEvaluation* evaluation) {
    if (!evaluation) return;
    
    if (evaluation->strengths) {
        for (size_t i = 0; i < evaluation->strength_count; i++) {
            if (evaluation->strengths[i]) {
                safe_free((void**)&evaluation->strengths[i]);
            }
        }
        safe_free((void**)&evaluation->strengths);
    }
    
    if (evaluation->weaknesses) {
        for (size_t i = 0; i < evaluation->weakness_count; i++) {
            if (evaluation->weaknesses[i]) {
                safe_free((void**)&evaluation->weaknesses[i]);
            }
        }
        safe_free((void**)&evaluation->weaknesses);
    }
    
    if (evaluation->recommendations) {
        for (size_t i = 0; i < evaluation->recommendation_count; i++) {
            if (evaluation->recommendations[i]) {
                safe_free((void**)&evaluation->recommendations[i]);
            }
        }
        safe_free((void**)&evaluation->recommendations);
    }
    
    safe_free((void**)&evaluation);
}

/**
 * @brief 优化产品设计（多因子加权优化算法）
 */
int optimize_product_design(ProductDesignEngine* engine,
                            ProductSpec* spec,
                            const DesignEvaluation* evaluation) {
    if (!engine || !spec || !evaluation) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    // 多因子加权优化：基于所有评估维度调整规格参数
    // 权重配置
    const double feasibility_w = 0.35;
    const double cost_w = 0.25;
    const double innovation_w = 0.20;
    const double market_w = 0.20;
    
    // 计算综合评分
    double composite = feasibility_w * evaluation->feasibility +
                       cost_w * evaluation->cost_effectiveness +
                       innovation_w * evaluation->innovation_level +
                       market_w * evaluation->market_potential;
    
    // 基于综合评分调整参数
    if (composite < 0.3) {
        // 评分极低：大幅度调整
        double factor = 0.6 + 0.3 * composite;
        spec->complexity_score *= factor;
        spec->estimated_cost *= factor;
        spec->development_time *= factor;
        spec->feasibility_score += 0.15;
    } else if (composite < 0.5) {
        // 评分较低：中度调整
        double factor = 0.75 + 0.2 * composite;
        spec->complexity_score *= factor;
        spec->estimated_cost *= factor;
        spec->development_time *= factor;
        spec->feasibility_score += 0.1;
    } else if (composite < 0.7) {
        // 评分中等：微调
        spec->complexity_score *= 0.95;
        spec->estimated_cost *= 0.95;
        spec->feasibility_score += 0.05;
    }
    
    // 各维度专项优化
    if (evaluation->feasibility < 0.4) {
        // 可行性差：降低复杂度以提升可行性
        spec->complexity_score *= 0.75;
        spec->feasibility_score += 0.2;
    }
    
    if (evaluation->cost_effectiveness < 0.4) {
        // 成本效益差：降低预估成本
        spec->estimated_cost *= 0.7;
        // 适当延长开发时间以降低成本
        spec->development_time *= 1.1;
    }
    
    if (evaluation->innovation_level < 0.3) {
        // 创新度低：适当增加复杂度以提升创新空间
        spec->complexity_score *= 1.15;
    }
    
    if (evaluation->market_potential < 0.3) {
        // 市场潜力低：降低预估成本以提升竞争力
        spec->estimated_cost *= 0.8;
    }
    
    // 参数边界限制
    if (spec->complexity_score < 0.1) spec->complexity_score = 0.1;
    if (spec->complexity_score > 10.0) spec->complexity_score = 10.0;
    if (spec->estimated_cost < 100.0) spec->estimated_cost = 100.0;
    if (spec->development_time < 0.5) spec->development_time = 0.5;
    if (spec->feasibility_score > 1.0) spec->feasibility_score = 1.0;
    if (spec->feasibility_score < 0.1) spec->feasibility_score = 0.1;
    
    return 0;
}

/**
 * @brief 自我改进：基于反馈迭代优化设计
 */
ProductSpec* self_improve_design(ProductDesignEngine* engine,
                                 const ProductRequirement* requirement,
                                 int iterations) {
    if (!engine || !requirement || iterations <= 0) {
        return NULL;
    }
    
    // 生成初始规格
    ProductSpec* spec = generate_product_spec(engine, requirement);
    if (!spec) return NULL;
    
    // 迭代优化
    for (int i = 0; i < iterations; i++) {
        // 评估当前设计
        DesignEvaluation* evaluation = evaluate_product_design(engine, spec);
        if (!evaluation) {
            // 评估失败，返回当前规格
            break;
        }
        
        // 优化设计
        int result = optimize_product_design(engine, spec, evaluation);
        design_evaluation_destroy(evaluation);
        
        if (result != 0) {
            break;
        }
    }
    
    return spec;
}

/* ============================================================================
 * F-10: 参数化生成设计实现
 * =========================================================================== */

/**
 * @brief 创建设计参数
 */
DesignParameter create_design_parameter(const char* name, DesignParamType param_type,
                                        double min_value, double max_value, double step) {
    DesignParameter param;
    memset(&param, 0, sizeof(DesignParameter));
    param.name = name ? string_duplicate_nullable(name) : string_duplicate_nullable("param");
    param.param_type = param_type;
    param.min_value = min_value;
    param.max_value = max_value;
    param.step = step;
    param.current_value = min_value;
    return param;
}

/**
 * @brief 计算每个参数的离散值数量
 */
static void param_discrete_counts(const DesignParameter* parameters, size_t param_count,
                                   size_t* counts) {
    for (size_t i = 0; i < param_count; i++) {
        if (parameters[i].step <= 0.0 || parameters[i].max_value <= parameters[i].min_value) {
            counts[i] = 1;
        } else {
            counts[i] = (size_t)((parameters[i].max_value - parameters[i].min_value) / parameters[i].step) + 1;
            if (counts[i] > 100) counts[i] = 100;
        }
    }
}

/**
 * @brief 设置参数数组到指定索引组合（笛卡尔积映射）
 */
static void param_set_by_index(DesignParameter* params, size_t param_count,
                                const size_t* counts, size_t flat_idx) {
    size_t divisor = 1;
    for (size_t i = 0; i < param_count; i++) {
        size_t idx = (flat_idx / divisor) % counts[i];
        double val = params[i].min_value + (double)idx * params[i].step;
        if (val > params[i].max_value) val = params[i].max_value;
        if (params[i].param_type == PARAM_TYPE_INT)
            val = (double)((int)(val + 0.5));
        else if (params[i].param_type == PARAM_TYPE_BOOL)
            val = (idx == 0) ? 0.0 : 1.0;
        params[i].current_value = val;
        divisor *= counts[i];
    }
}

/**
 * @brief 基于参数网格生成设计变体
 */
DesignVariant* generate_design_variants(ProductDesignEngine* engine,
                                        const ProductRequirement* requirement,
                                        const DesignParameter* parameters,
                                        size_t param_count,
                                        size_t* variant_count) {
    if (!engine || !requirement || !parameters || param_count == 0 || !variant_count) {
        return NULL;
    }
    
    size_t* counts = (size_t*)safe_calloc(param_count, sizeof(size_t));
    if (!counts) return NULL;
    param_discrete_counts(parameters, param_count, counts);
    
    size_t total = 1;
    for (size_t i = 0; i < param_count; i++) {
        total *= counts[i];
        if (total > 100000) { total = 100000; break; }
    }
    
    if (total == 0) { safe_free((void**)&counts); *variant_count = 0; return NULL; }
    
    DesignVariant* variants = (DesignVariant*)safe_calloc(total, sizeof(DesignVariant));
    if (!variants) { safe_free((void**)&counts); *variant_count = 0; return NULL; }
    
    size_t actual_count = 0;
    for (size_t fi = 0; fi < total; fi++) {
        DesignParameter* param_copy = (DesignParameter*)safe_calloc(param_count, sizeof(DesignParameter));
        if (!param_copy) break;
        for (size_t p = 0; p < param_count; p++) {
            param_copy[p] = parameters[p];
            param_copy[p].name = string_duplicate_nullable(parameters[p].name);
        }
        param_set_by_index(param_copy, param_count, counts, fi);
        
        char req_buf[4096] = {0};
        size_t pos = 0;
        if (requirement->requirement_text) {
            pos += snprintf(req_buf + pos, sizeof(req_buf) - pos, "%s ", requirement->requirement_text);
        }
        for (size_t p = 0; p < param_count; p++) {
            pos += snprintf(req_buf + pos, sizeof(req_buf) - pos, "%s=%.2g ", param_copy[p].name, param_copy[p].current_value);
        }
        
        ProductRequirement* var_req = parse_product_requirement(engine, req_buf);
        if (var_req) {
            variants[actual_count].spec = generate_product_spec(engine, var_req);
            if (variants[actual_count].spec) {
                variants[actual_count].evaluation = evaluate_product_design(engine, variants[actual_count].spec);
                if (variants[actual_count].evaluation) {
                    double cost_ratio = variants[actual_count].spec->estimated_cost / 100000.0;
                    if (cost_ratio > 1.0) cost_ratio = 1.0;
                    variants[actual_count].composite_score =
                        0.40 * variants[actual_count].evaluation->feasibility +
                        0.30 * (1.0 - cost_ratio) +
                        0.30 * variants[actual_count].evaluation->innovation_level;
                }
            }
            variants[actual_count].parameters = param_copy;
            variants[actual_count].param_count = param_count;
            actual_count++;
            product_requirement_destroy(var_req);
        } else {
            for (size_t p = 0; p < param_count; p++)
                safe_free((void**)&param_copy[p].name);
            safe_free((void**)&param_copy);
        }
    }
    
    safe_free((void**)&counts);
    *variant_count = actual_count;
    return variants;
}

/**
 * @brief 检查设计参数是否满足所有约束
 */
int constraint_satisfaction_check(const DesignParameter* parameters,
                                  size_t param_count,
                                  const ConstraintSpec* constraints,
                                  size_t constraint_count,
                                  size_t* violated_idx) {
    if (!parameters || !constraints) return 1;
    if (violated_idx) *violated_idx = constraint_count;
    
    for (size_t c = 0; c < constraint_count; c++) {
        double val = 0.0;
        for (size_t i = 0; i < constraints[c].coefficient_count && i < param_count; i++) {
            val += constraints[c].coefficients[i] * parameters[i].current_value;
        }
        int satisfied = 0;
        switch (constraints[c].type) {
            case CONSTRAINT_LINEAR_LE: satisfied = (val <= constraints[c].rhs + 1e-9); break;
            case CONSTRAINT_LINEAR_GE: satisfied = (val >= constraints[c].rhs - 1e-9); break;
            case CONSTRAINT_LINEAR_EQ: satisfied = (fabs(val - constraints[c].rhs) < 1e-9); break;
            case CONSTRAINT_BOUND: satisfied = 1; break;
        }
        if (!satisfied) {
            if (violated_idx) *violated_idx = c;
            return 0;
        }
    }
    return 1;
}

/**
 * @brief 设计变体排序比较函数（综合评分降序）
 */
static int variant_compare_desc(const void* a, const void* b) {
    const DesignVariant* va = (const DesignVariant*)a;
    const DesignVariant* vb = (const DesignVariant*)b;
    if (vb->composite_score > va->composite_score) return 1;
    if (vb->composite_score < va->composite_score) return -1;
    return 0;
}

/**
 * @brief 对设计变体按综合评分排序
 */
void rank_design_variants(DesignVariant* variants, size_t variant_count) {
    if (!variants || variant_count == 0) return;
    qsort(variants, variant_count, sizeof(DesignVariant), variant_compare_desc);
}

/**
 * @brief 销毁设计变体及其内部资源
 */
void design_variant_destroy(DesignVariant* variant) {
    if (!variant) return;
    if (variant->parameters) {
        for (size_t i = 0; i < variant->param_count; i++) {
            if (variant->parameters[i].name)
                safe_free((void**)&variant->parameters[i].name);
        }
        safe_free((void**)&variant->parameters);
    }
    if (variant->spec) product_spec_destroy(variant->spec);
    if (variant->evaluation) design_evaluation_destroy(variant->evaluation);
}

/* ============================================================================
 * F-10: 拓扑优化实现（SIMP方法）
 * =========================================================================== */

/**
 * @brief 构建4x4单元刚度矩阵（各向同性线弹性材料，平面应力）
 * 
 * 使用2x2高斯积分计算4节点四边形单元的刚度矩阵。
 * E=1, nu=0.3。
 */
static void build_element_stiffness_matrix(double* ke) {
    if (!ke) return;
    double E = 1.0;
    double nu = 0.3;
    double k[8][8];
    double B[3][8];
    double D[3][3];
    double pts[4][2] = {{-0.577350269189626, -0.577350269189626},
                        { 0.577350269189626, -0.577350269189626},
                        { 0.577350269189626,  0.577350269189626},
                        {-0.577350269189626,  0.577350269189626}};
    
    D[0][0] = E / (1.0 - nu * nu); D[0][1] = D[0][0] * nu; D[0][2] = 0.0;
    D[1][0] = D[0][1]; D[1][1] = D[0][0]; D[1][2] = 0.0;
    D[2][0] = 0.0; D[2][1] = 0.0; D[2][2] = E / (2.0 * (1.0 + nu));
    
    memset(k, 0, sizeof(k));
    
    for (int gp = 0; gp < 4; gp++) {
        double xi = pts[gp][0];
        double eta = pts[gp][1];
        
        double dN_dxi[4] = {
            -0.25 * (1.0 - eta),  0.25 * (1.0 - eta),
             0.25 * (1.0 + eta), -0.25 * (1.0 + eta)
        };
        double dN_deta[4] = {
            -0.25 * (1.0 - xi), -0.25 * (1.0 + xi),
             0.25 * (1.0 + xi),  0.25 * (1.0 - xi)
        };
        
        double J[2][2] = {{0,0},{0,0}};
        double node_coords[4][2] = {{-1,-1},{1,-1},{1,1},{-1,1}};
        for (int i = 0; i < 4; i++) {
            J[0][0] += dN_dxi[i] * node_coords[i][0];
            J[0][1] += dN_dxi[i] * node_coords[i][1];
            J[1][0] += dN_deta[i] * node_coords[i][0];
            J[1][1] += dN_deta[i] * node_coords[i][1];
        }
        double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        double invJ[2][2] = {{ J[1][1]/detJ, -J[0][1]/detJ},
                             {-J[1][0]/detJ,  J[0][0]/detJ}};
        
        double dN_dx[4], dN_dy[4];
        for (int i = 0; i < 4; i++) {
            dN_dx[i] = invJ[0][0] * dN_dxi[i] + invJ[0][1] * dN_deta[i];
            dN_dy[i] = invJ[1][0] * dN_dxi[i] + invJ[1][1] * dN_deta[i];
        }
        
        memset(B, 0, sizeof(B));
        for (int i = 0; i < 4; i++) {
            B[0][2*i]   = dN_dx[i];
            B[1][2*i+1] = dN_dy[i];
            B[2][2*i]   = dN_dy[i];
            B[2][2*i+1] = dN_dx[i];
        }
        
        double DBT[3][8] = {{0}};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 8; j++)
                for (int kk = 0; kk < 3; kk++)
                    DBT[i][j] += D[i][kk] * B[kk][j];
        
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                for (int kk = 0; kk < 3; kk++)
                    k[i][j] += B[kk][i] * DBT[kk][j] * detJ;
    }
    
    /* R6-003修复: 保留完整8x8单元刚度矩阵（含x+y方向DOF），而非仅提取x方向 */
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            ke[i * 8 + j] = k[i][j];
}

/**
 * @brief 创建拓扑优化状态
 */
TopologyOptimizationState* topology_optimization_create(int nelx, int nely,
                                                         double volfrac,
                                                         double penal,
                                                         double rmin) {
    /* P2修复: 添加上限检查，防止nelx*nely整数溢出导致分配极小内存 */
    if (nelx <= 0 || nely <= 0 || nelx > 10000 || nely > 10000 || volfrac <= 0.0 || volfrac > 1.0 || penal <= 0.0 || rmin <= 0.0) {
        return NULL;
    }
    
    TopologyOptimizationState* state = (TopologyOptimizationState*)safe_calloc(1, sizeof(TopologyOptimizationState));
    if (!state) return NULL;
    
    int n = nelx * nely;
    state->nelx = nelx;
    state->nely = nely;
    state->volfrac = volfrac;
    state->penal = penal;
    state->rmin = rmin;
    state->max_iterations = 200;
    state->change_threshold = 0.01;
    
    state->densities = (double*)safe_calloc(n, sizeof(double));
    state->densities_new = (double*)safe_calloc(n, sizeof(double));
    state->compliance = (double*)safe_calloc(n, sizeof(double));
    state->sensitivity = (double*)safe_calloc(n, sizeof(double));
    state->sensitivity_filter = (double*)safe_calloc(n, sizeof(double));
    state->ke = (double*)safe_calloc(64, sizeof(double)); /* R6-003修复: 8x8完整单元刚度矩阵 */
    
    if (!state->densities || !state->densities_new || !state->compliance ||
        !state->sensitivity || !state->sensitivity_filter || !state->ke) {
        topology_optimization_destroy(state);
        return NULL;
    }
    
    for (int i = 0; i < n; i++)
        state->densities[i] = volfrac;
    
    build_element_stiffness_matrix(state->ke);
    
    int ndof = 2 * (nelx + 1) * (nely + 1);
    state->f = (double*)safe_calloc(ndof, sizeof(double));
    if (!state->f) { topology_optimization_destroy(state); return NULL; }
    state->f[1] = -1.0;
    
    int fixed_count = 2 * (nely + 1);
    state->fixed_count = fixed_count;
    state->fixed_dofs = (int*)safe_calloc(fixed_count, sizeof(int));
    if (!state->fixed_dofs) { topology_optimization_destroy(state); return NULL; }
    for (int i = 0; i <= nely; i++) {
        state->fixed_dofs[2 * i] = 2 * i * (nelx + 1);
        state->fixed_dofs[2 * i + 1] = 2 * i * (nelx + 1) + 1;
    }
    
    return state;
}

/**
 * @brief 销毁拓扑优化状态
 */
void topology_optimization_destroy(TopologyOptimizationState* state) {
    if (!state) return;
    if (state->densities) safe_free((void**)&state->densities);
    if (state->densities_new) safe_free((void**)&state->densities_new);
    if (state->compliance) safe_free((void**)&state->compliance);
    if (state->sensitivity) safe_free((void**)&state->sensitivity);
    if (state->sensitivity_filter) safe_free((void**)&state->sensitivity_filter);
    if (state->ke) safe_free((void**)&state->ke);
    if (state->f) safe_free((void**)&state->f);
    if (state->fixed_dofs) safe_free((void**)&state->fixed_dofs);
    safe_free((void**)&state);
}

/**
 * @brief 求解稀疏线性系统 K * u = f (共轭梯度法)
 * 
 * 使用预处理共轭梯度法(PCG)求解有限元平衡方程。
 * 对角线预处理(雅可比预条件)。
 */
static int solve_cg(double* K, int* K_rows, int n_dof, double* f, double* u, int max_iter, double tol) {
    if (!K || !K_rows || !f || !u || n_dof <= 0) return -1;
    
    /* D12修复: 密集矩阵格式，K_rows[i]=n_dof为行步长 */
    int stride = K_rows[0];
    if (stride <= 0) stride = n_dof;
    
    double* r = (double*)safe_calloc(n_dof, sizeof(double));
    double* d = (double*)safe_calloc(n_dof, sizeof(double));
    double* q = (double*)safe_calloc(n_dof, sizeof(double));
    double* M = (double*)safe_calloc(n_dof, sizeof(double));
    if (!r || !d || !q || !M) {
        if (r) safe_free((void**)&r);
        if (d) safe_free((void**)&d);
        if (q) safe_free((void**)&q);
        if (M) safe_free((void**)&M);
        return -1;
    }
    
    /* Jacobi预处理器: 对角元倒数 */
    for (int i = 0; i < n_dof; i++) {
        M[i] = 1.0 / (K[i * stride + i] + 1e-12);
    }
    
    for (int i = 0; i < n_dof; i++) u[i] = 0.0;
    for (int i = 0; i < n_dof; i++) r[i] = f[i];
    for (int i = 0; i < n_dof; i++) d[i] = M[i] * r[i];
    
    double rho = 0.0;
    for (int i = 0; i < n_dof; i++) rho += r[i] * d[i];
    
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* 密集矩阵-向量乘法: q = K * d */
        for (int i = 0; i < n_dof; i++) {
            q[i] = 0.0;
            double* row = &K[i * stride];
            for (int j = 0; j < n_dof; j++) {
                q[i] += row[j] * d[j];
            }
        }
        
        double dq = 0.0;
        for (int i = 0; i < n_dof; i++) dq += d[i] * q[i];
        if (fabs(dq) < 1e-30) { iter++; break; }
        
        double alpha = rho / dq;
        double rho_new = 0.0;
        for (int i = 0; i < n_dof; i++) {
            u[i] += alpha * d[i];
            r[i] -= alpha * q[i];
        }
        for (int i = 0; i < n_dof; i++) {
            double s = M[i] * r[i];
            rho_new += r[i] * s;
        }
        
        double beta = rho_new / rho;
        for (int i = 0; i < n_dof; i++) d[i] = M[i] * r[i] + beta * d[i];
        
        double norm = 0.0;
        for (int i = 0; i < n_dof; i++) norm += r[i] * r[i];
        if (sqrt(norm) < tol) break;
        
        rho = rho_new;
    }
    
    safe_free((void**)&r);
    safe_free((void**)&d);
    safe_free((void**)&q);
    safe_free((void**)&M);
    return iter;
}

/**
 * @brief 执行单次SIMP拓扑优化迭代
 * 
 * 实现完整的SIMP方法单步迭代：
 * 1. 组装全局刚度矩阵（密度惩罚插值）
 * 2. 施加边界条件并求解有限元方程（PCG）
 * 3. 计算各单元柔顺度及灵敏度
 * 4. 灵敏度过滤（半径rmin内的加权平均）
 * 5. OC法更新密度
 */
double topology_optimization_simp_iteration(TopologyOptimizationState* state) {
    if (!state || !state->densities) return 0.0;
    
    int nelx = state->nelx, nely = state->nely;
    int n = nelx * nely;
    int ndof = 2 * (nelx + 1) * (nely + 1);
    
    /* D12修复: K矩阵分配为密集格式(ndof*ndof)而非稀疏格式(ndof*4)。
     * 装配代码使用密集索引K[edof[i]*ndof+edof[j]]，edof[i]可达ndof+3，
     * 稀疏格式下越界写入远超分配范围，导致ACCESS_VIOLATION(0xC0000005)。 */
    double* u = (double*)safe_calloc(ndof, sizeof(double));
    double* K = (double*)safe_calloc((size_t)ndof * ndof, sizeof(double));
    int* K_rows = (int*)safe_calloc(ndof, sizeof(int));
    if (!u || !K || !K_rows) {
        if (u) safe_free((void**)&u);
        if (K) safe_free((void**)&K);
        if (K_rows) safe_free((void**)&K_rows);
        return 0.0;
    }
    
    for (int i = 0; i < ndof; i++) {
        K_rows[i] = ndof;  /* 密集矩阵带宽=ndof */
    }
    
    for (int elx = 0; elx < nelx; elx++) {
        for (int ely = 0; ely < nely; ely++) {
            int idx = ely + elx * nely;
            double x = state->densities[idx];
            double x_penal = x > 0.01 ? pow(x, state->penal) : 0.0;
            
            int n1 = (nely + 1) * elx + ely;
            int n2 = (nely + 1) * (elx + 1) + ely;
            int edof[8] = {
                2 * n1, 2 * n1 + 1,
                2 * n2, 2 * n2 + 1,
                2 * n2 + 2, 2 * n2 + 3,
                2 * n1 + 2, 2 * n1 + 3
            };
            
            /* R6-003修复: 组装完整的8x8单元刚度矩阵到全局矩阵，而非仅对角 */
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    K[edof[i] * ndof + edof[j]] += x_penal * state->ke[i * 8 + j];
                }
            }
        }
    }
    
    for (int fi = 0; fi < state->fixed_count; fi++) {
        int dof = state->fixed_dofs[fi];
        if (dof < ndof) {
            for (int j = 0; j < ndof; j++) K[dof * ndof + j] = 0.0;
            K[dof * ndof + dof] = 1.0;
            u[dof] = 0.0;
        }
    }
    
    solve_cg(K, K_rows, ndof, state->f, u, 2000, 1e-12);
    
    state->total_compliance = 0.0;
    for (int elx = 0; elx < nelx; elx++) {
        for (int ely = 0; ely < nely; ely++) {
            int idx = ely + elx * nely;
            double x = state->densities[idx];
            double x_penal = x > 0.01 ? pow(x, state->penal) : 0.0;
            
            int n1 = (nely + 1) * elx + ely;
            int n2 = (nely + 1) * (elx + 1) + ely;
            int edof[8] = {
                2 * n1, 2 * n1 + 1,
                2 * n2, 2 * n2 + 1,
                2 * n2 + 2, 2 * n2 + 3,
                2 * n1 + 2, 2 * n1 + 3
            };
            
            double ue[8];
            for (int i = 0; i < 8; i++) ue[i] = u[edof[i]];
            
            double ce = 0.0;
            for (int i = 0; i < 4; i++) {
                double sum = 0.0;
                for (int j = 0; j < 4; j++)
                    sum += state->ke[i * 4 + j] * ue[2 * j];
                ce += ue[2 * i] * sum;
            }
            ce *= x_penal;
            
            state->compliance[idx] = ce;
            state->total_compliance += ce;
            state->sensitivity[idx] = -state->penal * x_penal / (x + 1e-12) * ce;
        }
    }
    
    double rmin = state->rmin;
    for (int i = 0; i < n; i++) state->sensitivity_filter[i] = 0.0;
    
    for (int elx = 0; elx < nelx; elx++) {
        for (int ely = 0; ely < nely; ely++) {
            int idx = ely + elx * nely;
            double sum_s = 0.0, sum_w = 0.0;
            int xmin = (int)fmax(elx - rmin, 0);
            int xmax = (int)fmin(elx + rmin, nelx - 1);
            int ymin = (int)fmax(ely - rmin, 0);
            int ymax = (int)fmin(ely + rmin, nely - 1);
            for (int kx = xmin; kx <= xmax; kx++) {
                for (int ky = ymin; ky <= ymax; ky++) {
                    int kidx = ky + kx * nely;
                    double w = fmax(0.0, rmin - sqrt((double)((elx - kx) * (elx - kx) + (ely - ky) * (ely - ky))));
                    sum_s += w * state->sensitivity[kidx];
                    sum_w += w;
                }
            }
            if (sum_w > 0) state->sensitivity_filter[idx] = sum_s / sum_w;
        }
    }
    
    double l1 = 0.0, l2 = 100000.0;
    double move = 0.2;
    double change = 0.0;
    
    for (int outer = 0; outer < 50; outer++) {
        double lmid = 0.5 * (l1 + l2);
        double vol = 0.0;
        double max_change = 0.0;
        
        for (int i = 0; i < n; i++) {
            double x = state->densities[i];
            double s = -state->sensitivity_filter[i] / (lmid + 1e-12);
            double xnew = s * x;
            if (xnew > x + move) xnew = x + move;
            if (xnew < x - move) xnew = x - move;
            if (xnew > 1.0) xnew = 1.0;
            if (xnew < 0.01) xnew = 0.01;
            state->densities_new[i] = xnew;
            vol += xnew;
            double ch = fabs(xnew - x);
            if (ch > max_change) max_change = ch;
        }
        
        change = max_change;
        vol /= (double)n;
        
        if (vol - state->volfrac > 1e-6) l1 = lmid;
        else l2 = lmid;
        
        if (fabs(vol - state->volfrac) < 1e-6) break;
    }
    
    for (int i = 0; i < n; i++)
        state->densities[i] = state->densities_new[i];
    
    safe_free((void**)&u);
    safe_free((void**)&K);
    safe_free((void**)&K_rows);
    
    state->iteration++;
    return change;
}

/**
 * @brief 运行完整拓扑优化流程
 */
int topology_optimization_run(TopologyOptimizationState* state) {
    if (!state) return SELFLNN_ERROR_INVALID_ARGUMENT;
    
    state->iteration = 0;
    
    for (int iter = 0; iter < state->max_iterations; iter++) {
        double change = topology_optimization_simp_iteration(state);
        
        if (change < state->change_threshold) {
            return 0;
        }
    }
    
    return state->iteration >= state->max_iterations ? 1 : 0;
}

/**
 * @brief 获取当前拓扑优化的密度分布
 */
const double* topology_optimization_get_densities(TopologyOptimizationState* state,
                                                   int* density_count) {
    if (!state || !density_count) return NULL;
    *density_count = state->nelx * state->nely;
    return state->densities;
}

/* 删除3个未声明且与公开API重复的孤立函数 */

/* ============================================================================
 *: 参考设计案例注入API
 * 为产品设计增强引擎提供跨领域工业设计先验案例的注入接口。
 * 在引擎初始化后通过此API将种子案例批量注入引擎内部案例库，
 * 供设计优化、类比推理和设计空间探索时作为参考基线。
 * ============================================================================ */

/**
 * @brief 向产品设计引擎注入参考设计案例
 *
 *修复: 实际将种子案例写入引擎内部的参考案例库，
 * 替代原来仅打印日志的空实现。每个案例包含名称、类别、特征标签和约束摘要，
 * 后续设计优化时可作为类比推理的参考基线。
 *
 * @param engine 产品设计引擎句柄
 * @param case_name 案例名称
 * @param category_name 类别名称
 * @param feature_tags 关键特征标签（逗号分隔）
 * @param constraint_summary 设计约束摘要
 * @param category 类别枚举值
 * @return 成功返回0，失败返回-1
 */
int product_design_engine_add_reference_case(ProductDesignEngine* engine,
    const char* case_name, const char* category_name,
    const char* feature_tags, const char* constraint_summary, int category)
{
    if (!engine || !case_name || !category_name) return -1;
    if (!engine->reference_cases) return -1;

    /* 动态扩容 */
    if (engine->reference_case_count >= engine->reference_case_capacity) {
        int new_capacity = engine->reference_case_capacity * 2;
        if (new_capacity > PDE_MAX_REFERENCE_CASES) {
            new_capacity = PDE_MAX_REFERENCE_CASES;
        }
        if (engine->reference_case_count >= new_capacity) return -1;

        ProductReferenceCase* new_cases = (ProductReferenceCase*)safe_realloc(
            engine->reference_cases, new_capacity * sizeof(ProductReferenceCase));
        if (!new_cases) return -1;
        engine->reference_cases = new_cases;
        engine->reference_case_capacity = new_capacity;
    }

    ProductReferenceCase* rc = &engine->reference_cases[engine->reference_case_count];
    memset(rc, 0, sizeof(ProductReferenceCase));

    strncpy(rc->name, case_name, sizeof(rc->name) - 1);
    strncpy(rc->category_name, category_name, sizeof(rc->category_name) - 1);
    strncpy(rc->feature_tags, feature_tags ? feature_tags : "", sizeof(rc->feature_tags) - 1);
    strncpy(rc->constraint_summary, constraint_summary ? constraint_summary : "", sizeof(rc->constraint_summary) - 1);
    rc->category = category;

    engine->reference_case_count++;
    return 0;
}

/**
 * @brief 获取引擎中已加载的参考案例数量
 *
 * @param engine 产品设计引擎句柄
 * @return 参考案例数量，失败返回0
 */
int product_design_engine_get_reference_case_count(ProductDesignEngine* engine)
{
    if (!engine) return 0;
    return engine->reference_case_count;
}