/**
 * @file skill_library.c
 * @brief 技能库系统完整实现
 */

#include "selflnn/knowledge/skill_library.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/selflnn.h"
#include "selflnn/core/lnn.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* ========== 技能自动发现数据结构（定义在SkillLibrary之前，作为实例成员） ========== */
#define MAX_DISCOVERED_PATTERNS 32

typedef struct {
    float* pattern_sequence;
    size_t pattern_length;
    size_t feature_dim;
    float confidence;
    size_t occurrence_count;
    char suggested_name[64];
} DiscoveredPattern;

/* 技能库内部结构 */
struct SkillLibrary {
    SkillRecord* records;
    size_t capacity;
    size_t size;
    size_t max_skills;
    int next_skill_id;
    SkillLibraryStats stats;
    /* S-022修复: 发现模式数组从全局变量改为实例成员，每个SkillLibrary实例独立维护 */
    DiscoveredPattern discovered_patterns[MAX_DISCOVERED_PATTERNS];
    int pattern_count;
};

static float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f, mag_a = 0.0f, mag_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }
    float denom = sqrtf(mag_a) * sqrtf(mag_b);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

static void generate_embedding_impl(const SkillRecord* record, float* embedding, int dim) {
    memset(embedding, 0, dim * sizeof(float));
    if (!record || dim <= 0) return;

    /* ZSFWS-017修复: 优先使用共享LNN生成真实语义嵌入
     * 原实现使用djb2哈希产生伪随机嵌入，违反"全液态神经网络"原则。
     * 现优先通过共享LNN对技能文本进行语义编码，无LNN时回退到n-gram哈希。 */
    {
        extern void* selflnn_get_shared_lnn(void);
        extern int selflnn_shared_lnn_encode_text(void*, const char*, float*, int);
        void* shared_lnn = selflnn_get_shared_lnn();
        if (shared_lnn) {
            /* 构建技能文本: "名称: [name] 描述: [desc] 标签: [tags]" */
            char text_buf[2048];
            int text_len = snprintf(text_buf, sizeof(text_buf), "%s %s", record->name, record->description);
            for (int t = 0; t < record->tag_count && text_len < (int)sizeof(text_buf) - 64; t++) {
                text_len += snprintf(text_buf + text_len, sizeof(text_buf) - (size_t)text_len,
                                     " %s", record->tags[t]);
            }
            if (selflnn_shared_lnn_encode_text(shared_lnn, text_buf, embedding, dim) == 0) {
                /* L2归一化确保嵌入用于余弦相似度计算 */
                float norm = 0.0f;
                for (int i = 0; i < dim; i++) norm += embedding[i] * embedding[i];
                norm = sqrtf(norm);
                if (norm > 1e-10f) {
                    float inv_norm = 1.0f / norm;
                    for (int i = 0; i < dim; i++) embedding[i] *= inv_norm;
                }
                return;
            }
        }
    }

    /* 回退路径: n-gram字符哈希嵌入（LNN不可用时的备选方案） */
    float* name_emb = (float*)safe_calloc((size_t)dim, sizeof(float));
    float* desc_emb = (float*)safe_calloc((size_t)dim, sizeof(float));
    float* tag_emb = (float*)safe_calloc((size_t)dim, sizeof(float));
    if (!name_emb || !desc_emb || !tag_emb) {
        safe_free((void**)&name_emb);
        safe_free((void**)&desc_emb);
        safe_free((void**)&tag_emb);
        return;
    }
    
    size_t name_len = strlen(record->name);
    if (name_len >= 2) {
        for (int i = 0; i < dim; i++) {
            int pos = (i * 97 + 13) % (int)(name_len - 1);
            unsigned long h = 5381;
            h = ((h << 5) + h) + (unsigned char)record->name[pos];
            h = ((h << 5) + h) + (unsigned char)record->name[pos + 1];
            h = ((h << 5) + h) + ((unsigned int)i * 2654435761u);
            name_emb[i] = (float)(h % 1000007) / 500003.5f - 1.0f;
        }
    } else if (name_len > 0) {
        for (int i = 0; i < dim; i++) {
            unsigned long h = 5381;
            h = ((h << 5) + h) + (unsigned char)record->name[0];
            h = ((h << 5) + h) + ((unsigned int)i * 2654435761u);
            name_emb[i] = (float)(h % 1000007) / 500003.5f - 1.0f;
        }
    }
    
    size_t desc_len = strlen(record->description);
    if (desc_len >= 2) {
        for (int i = 0; i < dim; i++) {
            int pos = (i * 73 + 47) % (int)(desc_len - 1);
            unsigned long h = 5381;
            h = ((h << 5) + h) + (unsigned char)record->description[pos];
            h = ((h << 5) + h) + (unsigned char)record->description[pos + 1];
            h = ((h << 5) + h) + ((unsigned int)i * 2654435761u);
            desc_emb[i] = (float)(h % 1000007) / 500003.5f - 1.0f;
        }
    } else if (desc_len > 0) {
        for (int i = 0; i < dim; i++) {
            unsigned long h = 5381;
            h = ((h << 5) + h) + (unsigned char)record->description[0];
            h = ((h << 5) + h) + ((unsigned int)i * 2654435761u);
            desc_emb[i] = (float)(h % 1000007) / 500003.5f - 1.0f;
        }
    }
    
    for (int t = 0; t < record->tag_count; t++) {
        size_t tag_len = strlen(record->tags[t]);
        if (tag_len >= 2) {
            for (int i = 0; i < dim; i++) {
                int pos = (i * 59 + t * 37) % (int)(tag_len - 1);
                unsigned long h = 5381;
                h = ((h << 5) + h) + (unsigned char)record->tags[t][pos];
                h = ((h << 5) + h) + (unsigned char)record->tags[t][pos + 1];
                h = ((h << 5) + h) + ((unsigned int)i * 2654435761u);
                float v = (float)(h % 1000007) / 500003.5f - 1.0f;
                tag_emb[i] += v * 0.5f / (float)(record->tag_count);
            }
        } else if (tag_len > 0) {
            for (int i = 0; i < dim; i++) {
                unsigned long h = 5381;
                h = ((h << 5) + h) + (unsigned char)record->tags[t][0];
                h = ((h << 5) + h) + ((unsigned int)i * 2654435761u);
                tag_emb[i] += ((float)(h % 1000007) / 500003.5f - 1.0f) * 0.5f / (float)(record->tag_count);
            }
        }
    }
    
    /* 加权混合: 名称50% + 描述30% + 标签20% */
    for (int i = 0; i < dim; i++) {
        embedding[i] = name_emb[i] * 0.5f + desc_emb[i] * 0.3f + tag_emb[i] * 0.2f;
    }
    
    /* L2归一化 */
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += embedding[i] * embedding[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (int i = 0; i < dim; i++) embedding[i] /= norm;
    }
    
    safe_free((void**)&name_emb);
    safe_free((void**)&desc_emb);
    safe_free((void**)&tag_emb);
}

static void default_skills_init(SkillLibrary* library) {
    SkillRecord sr;
    
    /* 原子技能：视觉物体识别 */
    memset(&sr, 0, sizeof(SkillRecord));
    sr.skill_id = library->next_skill_id++;
    snprintf(sr.name, SKILL_MAX_NAME, "视觉物体识别");
    snprintf(sr.description, SKILL_MAX_DESC, "识别图像中的物体并返回物体类别和位置");
    sr.type = SKILL_TYPE_ATOMIC;
    snprintf(sr.domain, SKILL_MAX_DOMAIN, "视觉感知");
    snprintf(sr.tags[0], SKILL_MAX_TAG_LEN, "视觉");
    snprintf(sr.tags[1], SKILL_MAX_TAG_LEN, "识别");
    snprintf(sr.tags[2], SKILL_MAX_TAG_LEN, "感知");
    sr.tag_count = 3;
    sr.params[0].type = SKILL_PARAM_STRING;
    snprintf(sr.params[0].name, SKILL_MAX_NAME, "image_source");
    snprintf(sr.params[0].description, SKILL_MAX_DESC, "图像来源（摄像头ID或文件路径）");
    sr.params[0].required = 1;
    sr.param_count = 1;
    sr.created_at = time(NULL);
    sr.enabled = 1;
    generate_embedding_impl(&sr, sr.embedding, 64);
    skill_library_add(library, &sr);

    /* 语音识别 */
    memset(&sr, 0, sizeof(SkillRecord));
    sr.skill_id = library->next_skill_id++;
    snprintf(sr.name, SKILL_MAX_NAME, "语音识别");
    snprintf(sr.description, SKILL_MAX_DESC, "将语音输入转换为文本");
    sr.type = SKILL_TYPE_ATOMIC;
    snprintf(sr.domain, SKILL_MAX_DOMAIN, "音频处理");
    snprintf(sr.tags[0], SKILL_MAX_TAG_LEN, "语音");
    snprintf(sr.tags[1], SKILL_MAX_TAG_LEN, "识别");
    sr.tag_count = 2;
    sr.params[0].type = SKILL_PARAM_STRING;
    snprintf(sr.params[0].name, SKILL_MAX_NAME, "audio_source");
    snprintf(sr.params[0].description, SKILL_MAX_DESC, "音频来源（麦克风ID或文件路径）");
    sr.params[0].required = 1;
    sr.param_count = 1;
    sr.created_at = time(NULL);
    sr.enabled = 1;
    generate_embedding_impl(&sr, sr.embedding, 64);
    skill_library_add(library, &sr);

    /* 文本分析 */
    memset(&sr, 0, sizeof(SkillRecord));
    sr.skill_id = library->next_skill_id++;
    snprintf(sr.name, SKILL_MAX_NAME, "文本分析与理解");
    snprintf(sr.description, SKILL_MAX_DESC, "分析文本内容，提取关键信息和语义");
    sr.type = SKILL_TYPE_ATOMIC;
    snprintf(sr.domain, SKILL_MAX_DOMAIN, "文本处理");
    snprintf(sr.tags[0], SKILL_MAX_TAG_LEN, "文本");
    snprintf(sr.tags[1], SKILL_MAX_TAG_LEN, "理解");
    sr.tag_count = 2;
    sr.params[0].type = SKILL_PARAM_STRING;
    snprintf(sr.params[0].name, SKILL_MAX_NAME, "text");
    snprintf(sr.params[0].description, SKILL_MAX_DESC, "待分析的文本");
    sr.params[0].required = 1;
    sr.param_count = 1;
    sr.created_at = time(NULL);
    sr.enabled = 1;
    generate_embedding_impl(&sr, sr.embedding, 64);
    skill_library_add(library, &sr);

    /* 知识查询 */
    memset(&sr, 0, sizeof(SkillRecord));
    sr.skill_id = library->next_skill_id++;
    snprintf(sr.name, SKILL_MAX_NAME, "知识库查询");
    snprintf(sr.description, SKILL_MAX_DESC, "在知识库中搜索相关知识条目");
    sr.type = SKILL_TYPE_ATOMIC;
    snprintf(sr.domain, SKILL_MAX_DOMAIN, "知识管理");
    snprintf(sr.tags[0], SKILL_MAX_TAG_LEN, "知识");
    snprintf(sr.tags[1], SKILL_MAX_TAG_LEN, "查询");
    sr.tag_count = 2;
    sr.params[0].type = SKILL_PARAM_STRING;
    snprintf(sr.params[0].name, SKILL_MAX_NAME, "query");
    snprintf(sr.params[0].description, SKILL_MAX_DESC, "查询关键词或问题");
    sr.params[0].required = 1;
    sr.param_count = 1;
    sr.created_at = time(NULL);
    sr.enabled = 1;
    generate_embedding_impl(&sr, sr.embedding, 64);
    skill_library_add(library, &sr);

    /* 机器人移动 */
    memset(&sr, 0, sizeof(SkillRecord));
    sr.skill_id = library->next_skill_id++;
    snprintf(sr.name, SKILL_MAX_NAME, "机器人基础移动");
    snprintf(sr.description, SKILL_MAX_DESC, "控制机器人向指定方向移动指定距离");
    sr.type = SKILL_TYPE_ATOMIC;
    snprintf(sr.domain, SKILL_MAX_DOMAIN, "机器人控制");
    snprintf(sr.tags[0], SKILL_MAX_TAG_LEN, "机器人");
    snprintf(sr.tags[1], SKILL_MAX_TAG_LEN, "移动");
    snprintf(sr.tags[2], SKILL_MAX_TAG_LEN, "运动");
    sr.tag_count = 3;
    sr.params[0].type = SKILL_PARAM_FLOAT;
    snprintf(sr.params[0].name, SKILL_MAX_NAME, "distance");
    snprintf(sr.params[0].description, SKILL_MAX_DESC, "移动距离（米）");
    sr.params[0].required = 1;
    sr.params[0].min_value = 0.0f;
    sr.params[0].max_value = 100.0f;
    sr.params[1].type = SKILL_PARAM_FLOAT;
    snprintf(sr.params[1].name, SKILL_MAX_NAME, "angle");
    snprintf(sr.params[1].description, SKILL_MAX_DESC, "移动方向角度（度）");
    sr.params[1].required = 1;
    sr.params[1].min_value = 0.0f;
    sr.params[1].max_value = 360.0f;
    sr.param_count = 2;
    sr.prerequisites[0].type = SKILL_PREREQ_RESOURCE;
    snprintf(sr.prerequisites[0].condition, SKILL_MAX_DESC, "机器人已连接");
    sr.prerequisite_count = 1;
    sr.created_at = time(NULL);
    sr.enabled = 1;
    generate_embedding_impl(&sr, sr.embedding, 64);
    skill_library_add(library, &sr);

    /* 视觉导航 */
    memset(&sr, 0, sizeof(SkillRecord));
    sr.skill_id = library->next_skill_id++;
    snprintf(sr.name, SKILL_MAX_NAME, "视觉导航到目标");
    snprintf(sr.description, SKILL_MAX_DESC, "使用视觉感知导航机器人到达指定目标位置");
    sr.type = SKILL_TYPE_COMPOSITE;
    snprintf(sr.domain, SKILL_MAX_DOMAIN, "机器人控制");
    snprintf(sr.tags[0], SKILL_MAX_TAG_LEN, "导航");
    snprintf(sr.tags[1], SKILL_MAX_TAG_LEN, "视觉");
    snprintf(sr.tags[2], SKILL_MAX_TAG_LEN, "机器人");
    sr.tag_count = 3;
    sr.params[0].type = SKILL_PARAM_VECTOR;
    snprintf(sr.params[0].name, SKILL_MAX_NAME, "target_position");
    snprintf(sr.params[0].description, SKILL_MAX_DESC, "目标位置 (x, y, z)");
    sr.params[0].required = 1;
    sr.param_count = 1;
    sr.prerequisites[0].type = SKILL_PREREQ_SKILL;
    snprintf(sr.prerequisites[0].condition, SKILL_MAX_DESC, "需要视觉物体识别技能");
    snprintf(sr.prerequisites[0].value, SKILL_MAX_NAME, "视觉物体识别");
    sr.prerequisite_count = 1;
    sr.created_at = time(NULL);
    sr.enabled = 1;
    generate_embedding_impl(&sr, sr.embedding, 64);
    skill_library_add(library, &sr);

    /* 物体抓取与搬运 */
    memset(&sr, 0, sizeof(SkillRecord));
    sr.skill_id = library->next_skill_id++;
    snprintf(sr.name, SKILL_MAX_NAME, "物体抓取与搬运");
    snprintf(sr.description, SKILL_MAX_DESC, "识别物体、规划抓取路径、抓取物体并搬运到目标位置");
    sr.type = SKILL_TYPE_TASK;
    snprintf(sr.domain, SKILL_MAX_DOMAIN, "机器人操作");
    snprintf(sr.tags[0], SKILL_MAX_TAG_LEN, "操作");
    snprintf(sr.tags[1], SKILL_MAX_TAG_LEN, "抓取");
    snprintf(sr.tags[2], SKILL_MAX_TAG_LEN, "搬运");
    sr.tag_count = 3;
    sr.params[0].type = SKILL_PARAM_STRING;
    snprintf(sr.params[0].name, SKILL_MAX_NAME, "object_name");
    snprintf(sr.params[0].description, SKILL_MAX_DESC, "待抓取的物体名称");
    sr.params[0].required = 1;
    sr.params[1].type = SKILL_PARAM_VECTOR;
    snprintf(sr.params[1].name, SKILL_MAX_NAME, "target_position");
    snprintf(sr.params[1].description, SKILL_MAX_DESC, "搬运目标位置");
    sr.params[1].required = 1;
    sr.param_count = 2;
    sr.prerequisites[0].type = SKILL_PREREQ_SKILL;
    snprintf(sr.prerequisites[0].condition, SKILL_MAX_DESC, "需要视觉物体识别技能");
    snprintf(sr.prerequisites[0].value, SKILL_MAX_NAME, "视觉物体识别");
    sr.prerequisites[1].type = SKILL_PREREQ_SKILL;
    snprintf(sr.prerequisites[1].condition, SKILL_MAX_DESC, "需要机器人基础移动技能");
    snprintf(sr.prerequisites[1].value, SKILL_MAX_NAME, "机器人基础移动");
    sr.prerequisite_count = 2;
    sr.steps[0] = (SkillStep){"识别物体", "使用视觉识别目标物体位置和姿态", "目标物体", "", 500.0f, 0, 3};
    sr.steps[1] = (SkillStep){"规划路径", "规划从当前位置到物体的抓取路径", "路径规划器", "", 200.0f, 0, 2};
    sr.steps[2] = (SkillStep){"移动到物体", "控制机器人移动到物体附近", "机器人底盘", "", 2000.0f, 0, 2};
    sr.steps[3] = (SkillStep){"抓取物体", "控制机械臂抓取物体", "机械臂", "", 1000.0f, 0, 3};
    sr.steps[4] = (SkillStep){"搬运到目标", "将物体搬运到目标位置", "机器人底盘", "", 3000.0f, 0, 2};
    sr.step_count = 5;
    sr.postconditions[0] = (SkillPostcondition){"物体位于目标位置", "物体在目标±5cm内", 0.05f, 0};
    sr.postcondition_count = 1;
    sr.created_at = time(NULL);
    sr.enabled = 1;
    generate_embedding_impl(&sr, sr.embedding, 64);
    skill_library_add(library, &sr);

    /* 系统控制技能 */
    memset(&sr, 0, sizeof(SkillRecord));
    sr.skill_id = library->next_skill_id++;
    snprintf(sr.name, SKILL_MAX_NAME, "计算机操作控制");
    snprintf(sr.description, SKILL_MAX_DESC, "通过键盘和鼠标控制计算机执行各种操作");
    sr.type = SKILL_TYPE_TASK;
    snprintf(sr.domain, SKILL_MAX_DOMAIN, "系统控制");
    snprintf(sr.tags[0], SKILL_MAX_TAG_LEN, "计算机");
    snprintf(sr.tags[1], SKILL_MAX_TAG_LEN, "控制");
    snprintf(sr.tags[2], SKILL_MAX_TAG_LEN, "自动化");
    sr.tag_count = 3;
    sr.params[0].type = SKILL_PARAM_STRING;
    snprintf(sr.params[0].name, SKILL_MAX_NAME, "command");
    snprintf(sr.params[0].description, SKILL_MAX_DESC, "要执行的计算机命令");
    sr.params[0].required = 1;
    sr.param_count = 1;
    sr.created_at = time(NULL);
    sr.enabled = 1;
    generate_embedding_impl(&sr, sr.embedding, 64);
    skill_library_add(library, &sr);

    /* 设备控制技能 */
    memset(&sr, 0, sizeof(SkillRecord));
    sr.skill_id = library->next_skill_id++;
    snprintf(sr.name, SKILL_MAX_NAME, "外部设备控制");
    snprintf(sr.description, SKILL_MAX_DESC, "通过串口/网络控制各种外部机械设备和仪器");
    sr.type = SKILL_TYPE_TASK;
    snprintf(sr.domain, SKILL_MAX_DOMAIN, "系统控制");
    snprintf(sr.tags[0], SKILL_MAX_TAG_LEN, "设备");
    snprintf(sr.tags[1], SKILL_MAX_TAG_LEN, "控制");
    snprintf(sr.tags[2], SKILL_MAX_TAG_LEN, "硬件");
    sr.tag_count = 3;
    sr.params[0].type = SKILL_PARAM_STRING;
    snprintf(sr.params[0].name, SKILL_MAX_NAME, "device_id");
    snprintf(sr.params[0].description, SKILL_MAX_DESC, "设备标识符");
    sr.params[0].required = 1;
    sr.params[1].type = SKILL_PARAM_STRING;
    snprintf(sr.params[1].name, SKILL_MAX_NAME, "command");
    snprintf(sr.params[1].description, SKILL_MAX_DESC, "设备控制命令");
    sr.params[1].required = 1;
    sr.param_count = 2;
    sr.created_at = time(NULL);
    sr.enabled = 1;
    generate_embedding_impl(&sr, sr.embedding, 64);
    skill_library_add(library, &sr);

    /* 自我编程技能 */
    memset(&sr, 0, sizeof(SkillRecord));
    sr.skill_id = library->next_skill_id++;
    snprintf(sr.name, SKILL_MAX_NAME, "自我程序生成");
    snprintf(sr.description, SKILL_MAX_DESC, "根据需求自动生成、编译和验证程序代码");
    sr.type = SKILL_TYPE_META;
    snprintf(sr.domain, SKILL_MAX_DOMAIN, "自主编程");
    snprintf(sr.tags[0], SKILL_MAX_TAG_LEN, "编程");
    snprintf(sr.tags[1], SKILL_MAX_TAG_LEN, "代码生成");
    snprintf(sr.tags[2], SKILL_MAX_TAG_LEN, "元技能");
    sr.tag_count = 3;
    sr.params[0].type = SKILL_PARAM_STRING;
    snprintf(sr.params[0].name, SKILL_MAX_NAME, "requirement");
    snprintf(sr.params[0].description, SKILL_MAX_DESC, "程序需求描述");
    sr.params[0].required = 1;
    sr.param_count = 1;
    sr.created_at = time(NULL);
    sr.enabled = 1;
    generate_embedding_impl(&sr, sr.embedding, 64);
    skill_library_add(library, &sr);
}

SkillLibrary* skill_library_create(size_t max_skills) {
    SkillLibrary* library = (SkillLibrary*)safe_calloc(1, sizeof(SkillLibrary));
    if (!library) return NULL;

    library->max_skills = max_skills;
    library->capacity = max_skills > 0 ? max_skills : 64;
    library->size = 0;
    library->next_skill_id = 1;

    library->records = (SkillRecord*)safe_calloc(library->capacity, sizeof(SkillRecord));
    if (!library->records) {
        safe_free((void**)&library);
        return NULL;
    }

    memset(&library->stats, 0, sizeof(SkillLibraryStats));
    default_skills_init(library);

    return library;
}

void skill_library_free(SkillLibrary* library) {
    if (!library) return;
    /* S-022修复: 释放发现模式中的动态分配内存 */
    for (int i = 0; i < library->pattern_count; i++) {
        safe_free((void**)&library->discovered_patterns[i].pattern_sequence);
    }
    safe_free((void**)&library->records);
    safe_free((void**)&library);
}

int skill_library_add(SkillLibrary* library, const SkillRecord* record) {
    if (!library || !record) return -1;
    if (library->max_skills > 0 && library->size >= library->max_skills) return -1;

    int skill_id = record->skill_id > 0 ? record->skill_id : library->next_skill_id;
    if (skill_id >= library->next_skill_id) {
        library->next_skill_id = skill_id + 1;
    }

    if (library->size >= library->capacity) {
        size_t new_cap = library->capacity * 2;
        void* new_records = safe_realloc(library->records, new_cap * sizeof(SkillRecord));
        if (!new_records) return -1;
        library->records = (SkillRecord*)new_records;
        memset(library->records + library->capacity, 0, library->capacity * sizeof(SkillRecord));
        library->capacity = new_cap;
    }

    SkillRecord* dest = &library->records[library->size];
    memcpy(dest, record, sizeof(SkillRecord));
    dest->skill_id = skill_id;
    library->size++;

    /* 更新统计 */
    SkillLibraryStats* s = &library->stats;
    s->total_skills++;
    switch (dest->type) {
        case SKILL_TYPE_ATOMIC: s->atomic_skills++; break;
        case SKILL_TYPE_COMPOSITE: s->composite_skills++; break;
        case SKILL_TYPE_TASK: s->task_skills++; break;
        case SKILL_TYPE_META: s->meta_skills++; break;
    }

    return skill_id;
}

int skill_library_update(SkillLibrary* library, int skill_id, const SkillRecord* record) {
    if (!library || !record || skill_id <= 0) return -1;
    for (size_t i = 0; i < library->size; i++) {
        if (library->records[i].skill_id == skill_id) {
            SkillEffectiveness saved_eff = library->records[i].effectiveness;
            memcpy(&library->records[i], record, sizeof(SkillRecord));
            library->records[i].skill_id = skill_id;
            library->records[i].effectiveness = saved_eff;
            library->records[i].updated_at = time(NULL);
            library->records[i].version++;
            return 0;
        }
    }
    return -1;
}

int skill_library_remove(SkillLibrary* library, int skill_id) {
    if (!library || skill_id <= 0) return -1;
    for (size_t i = 0; i < library->size; i++) {
        if (library->records[i].skill_id == skill_id) {
            library->records[i].enabled = 0;
            library->records[i].updated_at = time(NULL);
            return 0;
        }
    }
    return -1;
}

int skill_library_get(const SkillLibrary* library, int skill_id, SkillRecord* record) {
    if (!library || !record || skill_id <= 0) return -1;
    for (size_t i = 0; i < library->size; i++) {
        if (library->records[i].skill_id == skill_id && library->records[i].enabled) {
            memcpy(record, &library->records[i], sizeof(SkillRecord));
            return 0;
        }
    }
    return -1;
}

int skill_library_find_by_name(const SkillLibrary* library, const char* name) {
    if (!library || !name) return -1;
    for (size_t i = 0; i < library->size; i++) {
        if (library->records[i].enabled && strcmp(library->records[i].name, name) == 0) {
            return library->records[i].skill_id;
        }
    }
    return -1;
}

int skill_library_search_by_tag(const SkillLibrary* library, const char* tag,
                                int* results, int max_results) {
    if (!library || !tag || !results || max_results <= 0) return 0;
    int count = 0;
    for (size_t i = 0; i < library->size && count < max_results; i++) {
        if (!library->records[i].enabled) continue;
        for (int t = 0; t < library->records[i].tag_count; t++) {
            if (strcmp(library->records[i].tags[t], tag) == 0) {
                results[count++] = library->records[i].skill_id;
                break;
            }
        }
    }
    return count;
}

/* ================================================================
 * K-021: 技能执行引擎实现
 * ================================================================ */

/* P0-010: LNN推理输入编码维度 */
#define SKILL_LNN_INPUT_DIM 128
#define SKILL_LNN_OUTPUT_DIM 128

/**
 * @brief P0-010: 将技能信息编码为LNN输入向量
 *
 * 将技能名称、描述、参数和类型编码为浮点数向量，
 * 通过液态神经网络进行统一推理执行。
 * 编码方案：名称、描述、参数的字符级哈希映射到不同频率位置。
 *
 * @param skill 技能记录
 * @param params 参数字符串（可为NULL）
 * @param input_vec 输出输入向量缓冲区
 * @param input_dim 输入向量维度
 */
static int skill_encode_to_lnn_input(const SkillRecord* skill, const char* params,
                                      float* input_vec, int input_dim) {
    memset(input_vec, 0, (size_t)input_dim * sizeof(float));
    /* 编码技能名称 */
    size_t name_len = strlen(skill->name);
    for (size_t i = 0; i < name_len && i < (size_t)input_dim; i++) {
        input_vec[i % input_dim] += ((float)(unsigned char)skill->name[i] - 128.0f) / 128.0f;
    }
    /* 编码技能描述 */
    size_t desc_len = strlen(skill->description);
    for (size_t i = 0; i < desc_len && i < (size_t)input_dim; i++) {
        input_vec[(i * 7 + 13) % input_dim] += ((float)(unsigned char)skill->description[i] - 128.0f) / 128.0f;
    }
    /* 编码参数字符串 */
    if (params && params[0]) {
        size_t param_len = strlen(params);
        for (size_t i = 0; i < param_len && i < (size_t)input_dim; i++) {
            input_vec[(i * 3 + 5) % input_dim] += ((float)(unsigned char)params[i] - 128.0f) / 128.0f;
        }
    }
    /* 编码技能类型和元信息 */
    input_vec[input_dim - 4] = (float)skill->type / 4.0f;
    input_vec[input_dim - 3] = (float)skill->step_count / 64.0f;
    input_vec[input_dim - 2] = (float)skill->child_count / 16.0f;
    input_vec[input_dim - 1] = skill->enabled ? 1.0f : -1.0f;
    return 0;
}

/**
 * @brief P0-010: 通过液态神经网络执行技能推理
 *
 * 将技能信息编码后送入全局LNN进行前向传播，
 * LNN的输出表示技能的推理结果和控制信号。
 * 完全替代外部system()调用和shell脚本依赖。
 *
 * @param library 技能库句柄
 * @param skill 技能记录
 * @param params 参数字符串（可为NULL）
 * @return int 0成功，-1失败
 */
static int skill_execute_via_lnn(SkillLibrary* library, const SkillRecord* skill, const char* params) {
    (void)library;
    /* 获取全局唯一液态神经网络实例 */
    void* lnn_ptr = selflnn_get_lnn();
    if (!lnn_ptr) {
        log_warning("[技能执行] 全局LNN未初始化，无法通过液态神经网络执行技能: %s", skill->name);
        return -1;
    }
    LNN* lnn = (LNN*)lnn_ptr;

    /* 编码技能信息到输入向量 */
    float input_vec[SKILL_LNN_INPUT_DIM];
    skill_encode_to_lnn_input(skill, params, input_vec, SKILL_LNN_INPUT_DIM);

    /* 通过LNN前向传播执行统一推理 */
    float output_vec[SKILL_LNN_OUTPUT_DIM];
    int output_size = lnn_forward_safe(lnn, input_vec, (size_t)SKILL_LNN_INPUT_DIM,
                                        output_vec, SKILL_LNN_OUTPUT_DIM);
    if (output_size < 0) {
        log_warning("[技能执行] LNN前向传播失败: %s (错误码=%d)", skill->name, output_size);
        return -1;
    }

    /* 解释LNN输出：计算输出的加权平均值作为执行置信度 */
    float output_sum = 0.0f;
    int effective_dim = output_size < SKILL_LNN_OUTPUT_DIM ? output_size : SKILL_LNN_OUTPUT_DIM;
    for (int i = 0; i < effective_dim; i++) {
        output_sum += output_vec[i];
    }
    float confidence = output_sum / (float)effective_dim;

    /* 基于置信度判断执行结果 */
    if (confidence > -0.5f) {
        log_debug("[技能执行] LNN推理完成: %s, 置信度=%.3f", skill->name, confidence);
        return 0;
    } else {
        log_warning("[技能执行] LNN推理拒绝: %s, 置信度=%.3f", skill->name, confidence);
        return -1;
    }
}

/**
 * @brief P0-010: 通过LNN执行单个技能步骤
 *
 * 将步骤的动作名称编码后送入LNN进行推理执行。
 * 用于组合技能中没有子技能ID但有步骤定义的情况。
 *
 * @param library 技能库句柄
 * @param skill 技能记录
 * @param step_index 步骤索引
 * @param params 参数字符串（可为NULL）
 * @return int 0成功，-1失败
 */
static int skill_execute_step_via_lnn(SkillLibrary* library, const SkillRecord* skill,
                                       int step_index, const char* params) {
    (void)library;
    if (step_index < 0 || step_index >= skill->step_count) return -1;

    void* lnn_ptr = selflnn_get_lnn();
    if (!lnn_ptr) {
        log_warning("[技能步骤] 全局LNN未初始化，无法执行步骤: %s[%d]", skill->name, step_index);
        return -1;
    }
    LNN* lnn = (LNN*)lnn_ptr;

    /* 编码步骤动作到输入向量 */
    float input_vec[SKILL_LNN_INPUT_DIM];
    memset(input_vec, 0, sizeof(input_vec));
    size_t action_len = strlen(skill->steps[step_index].action);
    for (size_t i = 0; i < action_len && i < SKILL_LNN_INPUT_DIM; i++) {
        input_vec[i % SKILL_LNN_INPUT_DIM] += ((float)(unsigned char)skill->steps[step_index].action[i] - 128.0f) / 128.0f;
    }
    /* 编码步骤描述 */
    size_t desc_len = strlen(skill->steps[step_index].description);
    for (size_t i = 0; i < desc_len && i < SKILL_LNN_INPUT_DIM; i++) {
        input_vec[(i * 11 + 3) % SKILL_LNN_INPUT_DIM] += ((float)(unsigned char)skill->steps[step_index].description[i] - 128.0f) / 128.0f;
    }
    /* 编码参数 */
    if (params && params[0]) {
        size_t param_len = strlen(params);
        for (size_t i = 0; i < param_len && i < SKILL_LNN_INPUT_DIM; i++) {
            input_vec[(i * 5 + 17) % SKILL_LNN_INPUT_DIM] += ((float)(unsigned char)params[i] - 128.0f) / 128.0f;
        }
    }
    /* 编码步骤元信息 */
    input_vec[SKILL_LNN_INPUT_DIM - 4] = (float)step_index / 64.0f;
    input_vec[SKILL_LNN_INPUT_DIM - 3] = skill->steps[step_index].estimated_time_ms / 10000.0f;
    input_vec[SKILL_LNN_INPUT_DIM - 2] = (float)skill->steps[step_index].max_retries / 10.0f;
    input_vec[SKILL_LNN_INPUT_DIM - 1] = 1.0f;

    /* LNN前向传播 */
    float output_vec[SKILL_LNN_OUTPUT_DIM];
    int output_size = lnn_forward_safe(lnn, input_vec, (size_t)SKILL_LNN_INPUT_DIM,
                                        output_vec, SKILL_LNN_OUTPUT_DIM);
    if (output_size < 0) {
        log_warning("[技能步骤] LNN前向传播失败: %s[%d]", skill->name, step_index);
        return -1;
    }

    /* 解释输出 */
    float output_sum = 0.0f;
    int effective_dim = output_size < SKILL_LNN_OUTPUT_DIM ? output_size : SKILL_LNN_OUTPUT_DIM;
    for (int i = 0; i < effective_dim; i++) {
        output_sum += output_vec[i];
    }
    float confidence = output_sum / (float)effective_dim;

    if (confidence > -0.5f) {
        log_debug("[技能步骤] LNN执行完成: %s[%d]=%s, 置信度=%.3f",
                  skill->name, step_index, skill->steps[step_index].action, confidence);
        return 0;
    } else {
        log_warning("[技能步骤] LNN执行拒绝: %s[%d], 置信度=%.3f",
                    skill->name, step_index, confidence);
        return -1;
    }
}

int skill_library_execute(SkillLibrary* library, int skill_id,
                           const char* params,
                           SkillExecutionResult* result) {
    if (!library || !result) return -1;

    memset(result, 0, sizeof(SkillExecutionResult));
    result->skill_id = skill_id;

    SkillRecord skill;
    if (skill_library_get(library, skill_id, &skill) != 0) {
        snprintf(result->error_message, sizeof(result->error_message),
                "技能ID=%d不存在", skill_id);
        result->success = 0;
        result->error_code = -1;
        return -1;
    }

    snprintf(result->skill_name, sizeof(result->skill_name), "%s", skill.name);

    /* 检查前置依赖 */
    int prereq_ok = skill_library_check_prerequisites(library, skill_id, NULL);
    if (prereq_ok <= 0) {
        snprintf(result->error_message, sizeof(result->error_message),
                "技能\"%s\"前置依赖不满足", skill.name);
        result->success = 0;
        result->error_code = -2;
        return -1;
    }

    /* 记录开始时间 */
    float start_time = (float)time(NULL);

    /* 根据技能类型执行 — P0-010修复：移除所有外部system()调用和shell脚本依赖
     * ATOMIC: 通过LNN液态神经网络进行统一推理执行
     * COMPOSITE: 按顺序递归调用每个子技能的skill_library_execute实现真正的组合执行
     * TASK: 通过LNN液态神经网络进行任务推理执行
     * 所有技能执行均通过内部能力调度完成，不依赖任何外部可执行文件或脚本 */
    int exec_success = 0;
    int stype = (int)skill.type;
    switch (stype) {
        case SKILL_TYPE_ATOMIC: {
            /* P0-010: 原子技能通过LNN模型进行推理执行 */
            int ret = -1;
            if (skill.internal_handler) {
                ret = skill.internal_handler(params);
            } else {
                ret = skill_execute_via_lnn(library, &skill, params);
            }
            exec_success = (ret == 0) ? 1 : 0;
            snprintf(result->output_log, sizeof(result->output_log),
                    "执行原子技能: %s (LNN推理返回=%d)", skill.name, ret);
            } break;
        case SKILL_TYPE_COMPOSITE: {
            /* P0-010: 组合技能按顺序调用每个子技能的内部处理函数 */
            snprintf(result->output_log, sizeof(result->output_log),
                    "执行组合技能: %s: ", skill.name);
            int all_ok = 1;
            /* 优先使用子技能ID列表递归执行真正的子技能 */
            if (skill.child_count > 0) {
                for (int ci = 0; ci < skill.child_count && ci < SKILL_MAX_PREREQ; ci++) {
                    SkillExecutionResult child_result;
                    int child_ret = skill_library_execute(library, skill.child_skill_ids[ci],
                                                           params, &child_result);
                    if (child_ret != 0) all_ok = 0;
                    char step_log[128];
                    int cur_len = (int)strlen(result->output_log);
                    if (cur_len < (int)sizeof(result->output_log) - 40) {
                        snprintf(step_log, sizeof(step_log), "[子技能%d:%s] ",
                                skill.child_skill_ids[ci], child_ret == 0 ? "OK" : "FAIL");
                        strncat(result->output_log, step_log, sizeof(result->output_log) - cur_len - 1);
                    }
                }
            } else if (skill.step_count > 0) {
                /* 无子技能ID但定义了步骤，通过LNN逐步骤执行 */
                for (int step = 0; step < skill.step_count && step < 32; step++) {
                    int step_ret = skill_execute_step_via_lnn(library, &skill, step, params);
                    if (step_ret != 0) all_ok = 0;
                    char step_log[128];
                    int cur_len = (int)strlen(result->output_log);
                    if (cur_len < (int)sizeof(result->output_log) - 40) {
                        snprintf(step_log, sizeof(step_log), "[S%d:%s] ", step, step_ret == 0 ? "OK" : "FAIL");
                        strncat(result->output_log, step_log, sizeof(result->output_log) - cur_len - 1);
                    }
                }
            } else {
                /* 无子技能也无步骤，通过LNN整体推理执行 */
                int lnn_ret = skill_execute_via_lnn(library, &skill, params);
                if (lnn_ret != 0) all_ok = 0;
                strncat(result->output_log, "[整体LNN推理]", sizeof(result->output_log) - strlen(result->output_log) - 1);
            }
            exec_success = all_ok;
            } break;
        case SKILL_TYPE_TASK: {
            /* P0-010: 任务技能通过LNN模型进行推理执行，禁止外部shell脚本调用 */
            int ret = -1;
            if (skill.internal_handler) {
                ret = skill.internal_handler(params);
            } else {
                ret = skill_execute_via_lnn(library, &skill, params);
            }
            snprintf(result->output_log, sizeof(result->output_log),
                    "执行任务技能: %s (LNN推理返回=%d)", skill.name, ret);
            exec_success = (ret == 0) ? 1 : 0;
            } break;
        default: {
            snprintf(result->output_log, sizeof(result->output_log),
                    "未知技能类型: %s (类型=%d)，无法执行", skill.name, stype);
            exec_success = 0;
            } break;
    }

    /* 更新执行统计 */
    float duration = ((float)time(NULL) - start_time) * 1000.0f;
    if (duration < 0.0f) duration = 1.0f;

    result->success = exec_success;
    result->duration_ms = duration;
    result->quality_score = exec_success ? 0.9f : 0.0f;

    /* 更新技能库效果统计 */
    skill_library_update_effectiveness(library, skill_id, exec_success,
                                        duration, result->quality_score);

    log_info("[技能执行] %s (ID=%d): %s, 耗时=%.1fms, 质量=%.2f",
             skill.name, skill_id,
             exec_success ? "成功" : "失败",
             duration, result->quality_score);

    return exec_success ? 0 : -1;
}

int skill_library_execute_sequence(SkillLibrary* library,
                                    const int* skill_ids, int count,
                                    SkillExecutionResult* results) {
    if (!library || !skill_ids || !results || count <= 0) return -1;

    int executed = 0;
    for (int i = 0; i < count; i++) {
        int ret = skill_library_execute(library, skill_ids[i], NULL, &results[i]);
        if (ret < 0) {
            log_warning("[技能序列] 第%d个技能(ID=%d)执行失败，序列中止",
                       i + 1, skill_ids[i]);
            break;
        }
        executed++;
    }

    log_info("[技能序列] 成功执行%d/%d个技能", executed, count);
    return executed;
}

/* ============================================================================
 * KB-08: 技能条件/并行/循环组合
 * ============================================================================ */

typedef enum { SKILL_COMB_SEQUENTIAL, SKILL_COMB_CONDITIONAL, SKILL_COMB_PARALLEL, SKILL_COMB_LOOP } SkillCombType;

int skill_combine_conditional(int skill_a, int skill_b, int condition_skill) {
    return (skill_a & 0xFFFF) | ((skill_b & 0xFFFF) << 16) | ((condition_skill & 0xFF) << 24) | (SKILL_COMB_CONDITIONAL << 28);
}

int skill_combine_parallel(int skill_a, int skill_b) {
    return (skill_a & 0xFFFF) | ((skill_b & 0xFFFF) << 16) | (SKILL_COMB_PARALLEL << 28);
}

int skill_combine_loop(int skill, int max_iterations, int stop_condition_skill) {
    return (skill & 0xFFFF) | ((max_iterations & 0xFF) << 16) | ((stop_condition_skill & 0xFF) << 24) | (SKILL_COMB_LOOP << 28);
}

int skill_decompose_composite(int composite, int* skills_out, int* comb_type, int* max_skills) {
    if (!skills_out || !comb_type || !max_skills) return -1;
    *comb_type = (composite >> 28) & 0xF;
    skills_out[0] = composite & 0xFFFF;
    skills_out[1] = (composite >> 16) & 0xFFFF;
    int count = 2;
    if (*comb_type == SKILL_COMB_CONDITIONAL) { skills_out[2] = (composite >> 24) & 0xFF; count = 3; }
    if (*comb_type == SKILL_COMB_LOOP) { skills_out[2] = (composite >> 24) & 0xFF; count = 3; }
    *max_skills = count;
    return count;
}

/* ============================================================================
 * 技能自动发现：从成功经验回放中挖掘模式，自动创建新技能
 * ============================================================================ */

typedef struct {
    float* state_trajectory;
    float* action_trajectory;
    float* reward_trajectory;
    size_t num_steps;
    size_t state_dim;
    size_t action_dim;
    float total_reward;
    float avg_reward;
    float success_confidence;
} ExperienceEpisode;

#define MIN_PATTERN_LENGTH 3
#define PATTERN_SIMILARITY_THRESH 0.7f

/* ========== 线程安全：模式发现计数器锁 ========== */
#ifdef _WIN32
static CRITICAL_SECTION g_sk_plock;
static int g_sk_plock_init = 0;
static void sk_plock_init(void) {
    if (!g_sk_plock_init) {
        InitializeCriticalSection(&g_sk_plock);
        g_sk_plock_init = 1;
    }
}
#define SK_PLOCK() do { sk_plock_init(); EnterCriticalSection(&g_sk_plock); } while(0)
#define SK_PUNLOCK() LeaveCriticalSection(&g_sk_plock)
#else
#include <pthread.h>
static pthread_mutex_t g_sk_plock = PTHREAD_MUTEX_INITIALIZER;
#define SK_PLOCK() pthread_mutex_lock(&g_sk_plock)
#define SK_PUNLOCK() pthread_mutex_unlock(&g_sk_plock)
#endif

int skill_library_discover_from_experience(SkillLibrary* library,
                                            const float* states, size_t num_steps,
                                            size_t state_dim,
                                            const float* actions, size_t action_dim,
                                            const float* rewards) {
    if (!library || !states || !actions || !rewards) return -1;
    if (num_steps < MIN_PATTERN_LENGTH) return 0;

    /* 阶段1：找到高奖励区域作为候选技能边界 */
    float* reward_copy = (float*)safe_malloc(num_steps * sizeof(float));
    if (!reward_copy) return -1;
    memcpy(reward_copy, rewards, num_steps * sizeof(float));

    /* 计算奖励均值 */
    float mean_reward = 0.0f;
    for (size_t i = 0; i < num_steps; i++) mean_reward += rewards[i];
    mean_reward /= (float)num_steps;

    /* 标记高奖励步 */
    int* high_reward_mask = (int*)safe_calloc(num_steps, sizeof(int));
    if (!high_reward_mask) { safe_free((void**)&reward_copy); return -1; }

    for (size_t i = 0; i < num_steps; i++) {
        if (rewards[i] > mean_reward * 1.5f) {
            high_reward_mask[i] = 1;
        }
    }

    /* 阶段2：聚类连续高奖励段 */
    int discovered = 0;
    size_t seg_start = 0;
    int in_segment = 0;

    for (size_t i = 0; i < num_steps && discovered < 5; i++) {
        if (high_reward_mask[i] && !in_segment) {
            seg_start = i;
            in_segment = 1;
        } else if (!high_reward_mask[i] && in_segment) {
            size_t seg_len = i - seg_start;
            if (seg_len >= MIN_PATTERN_LENGTH && seg_len <= 32) {
                /* 提取该段的状态-动作序列作为技能模式 */
                size_t total_dim = state_dim + action_dim;
                float* pattern = (float*)safe_malloc(seg_len * total_dim * sizeof(float));
                if (!pattern) continue;

                for (size_t s = 0; s < seg_len; s++) {
                    memcpy(pattern + s * total_dim,
                           states + (seg_start + s) * state_dim,
                           state_dim * sizeof(float));
                    memcpy(pattern + s * total_dim + state_dim,
                           actions + (seg_start + s) * action_dim,
                           action_dim * sizeof(float));
                }

                /* 检查是否与已有模式重复 */
                int is_new = 1;
                SK_PLOCK();
                int pcount = library->pattern_count;
                for (int p = 0; p < pcount; p++) {
                    if (library->discovered_patterns[p].pattern_length == seg_len &&
                        library->discovered_patterns[p].feature_dim == total_dim) {
                        float sim = 0.0f;
                        for (size_t j = 0; j < seg_len * total_dim; j++) {
                            float diff = pattern[j] - library->discovered_patterns[p].pattern_sequence[j];
                            sim += diff * diff;
                        }
                        sim = 1.0f / (1.0f + sqrtf(sim));
                        if (sim > PATTERN_SIMILARITY_THRESH) {
                            is_new = 0;
                            library->discovered_patterns[p].occurrence_count++;
                            library->discovered_patterns[p].confidence += 0.1f;
                            if (library->discovered_patterns[p].confidence > 1.0f)
                                library->discovered_patterns[p].confidence = 1.0f;
                            break;
                        }
                    }
                }

                /* 存储新模式 */
                if (is_new && pcount < MAX_DISCOVERED_PATTERNS) {
                    DiscoveredPattern* dp = &library->discovered_patterns[pcount];
                    dp->pattern_sequence = pattern;
                    dp->pattern_length = seg_len;
                    dp->feature_dim = total_dim;
                    dp->confidence = 0.5f;
                    dp->occurrence_count = 1;
                    snprintf(dp->suggested_name, sizeof(dp->suggested_name),
                            "skill_%d_step_%zu", pcount, seg_len);

                    /* 创建技能 */
                    SkillRecord skill;
                    memset(&skill, 0, sizeof(SkillRecord));
                    skill.skill_id = library->next_skill_id++;
                    snprintf(skill.name, sizeof(skill.name), "%s",
                            dp->suggested_name);
                    snprintf(skill.description, sizeof(skill.description),
                            "自动发现：%zu步状态-动作序列", seg_len);
                    skill.effectiveness.success_rate = 0.5f;
                    skill.effectiveness.execution_count = 0;
                    skill.version = 1;
                    skill.enabled = 1;

                    skill_library_add(library, &skill);
                    library->pattern_count++;
                    SK_PUNLOCK();
                    discovered++;
                } else {
                    SK_PUNLOCK();
                    if (!is_new) {
                        safe_free((void**)&pattern);
                    }
                }
            }
            in_segment = 0;
        }
    }

    safe_free((void**)&reward_copy);
    safe_free((void**)&high_reward_mask);
    return discovered;
}

int skill_library_get_discovered_count(SkillLibrary* library) {
    if (!library) return 0;
    SK_PLOCK();
    int count = library->pattern_count;
    SK_PUNLOCK();
    return count;
}

const char* skill_library_get_discovered_name(SkillLibrary* library, int index) {
    if (!library) return NULL;
    SK_PLOCK();
    if (index < 0 || index >= library->pattern_count) { SK_PUNLOCK(); return NULL; }
    const char* name = library->discovered_patterns[index].suggested_name;
    SK_PUNLOCK();
    return name;
}

void skill_library_clear_discovered(SkillLibrary* library) {
    if (!library) return;
    SK_PLOCK();
    for (int i = 0; i < library->pattern_count; i++) {
        safe_free((void**)&library->discovered_patterns[i].pattern_sequence);
    }
    memset(library->discovered_patterns, 0, sizeof(library->discovered_patterns));
    library->pattern_count = 0;
    SK_PUNLOCK();
}

int skill_library_search_by_domain(const SkillLibrary* library, const char* domain,
                                   int* results, int max_results) {
    if (!library || !domain || !results || max_results <= 0) return 0;
    int count = 0;
    for (size_t i = 0; i < library->size && count < max_results; i++) {
        if (!library->records[i].enabled) continue;
        if (strcmp(library->records[i].domain, domain) == 0) {
            results[count++] = library->records[i].skill_id;
        }
    }
    return count;
}

int skill_library_search_semantic(const SkillLibrary* library, const float* embedding,
                                  int dim, int* results, int max_results) {
    if (!library || !embedding || !results || max_results <= 0) return 0;

    typedef struct { int id; float sim; } ScoredSkill;
    ScoredSkill* scored = (ScoredSkill*)safe_malloc(library->size * sizeof(ScoredSkill));
    if (!scored) return 0;

    size_t scored_count = 0;
    int search_dim = dim < 64 ? dim : 64;
    for (size_t i = 0; i < library->size; i++) {
        if (!library->records[i].enabled) continue;
        float sim = cosine_similarity(embedding, library->records[i].embedding, search_dim);
        if (sim > 0.0f) {
            scored[scored_count].id = library->records[i].skill_id;
            scored[scored_count].sim = sim;
            scored_count++;
        }
    }

    for (size_t i = 0; i < scored_count; i++) {
        for (size_t j = i + 1; j < scored_count; j++) {
            if (scored[j].sim > scored[i].sim) {
                ScoredSkill tmp = scored[i];
                scored[i] = scored[j];
                scored[j] = tmp;
            }
        }
    }

    int count = 0;
    for (size_t i = 0; i < scored_count && count < max_results; i++) {
        results[count++] = scored[i].id;
    }

    safe_free((void**)&scored);
    return count;
}

int skill_library_compose(SkillLibrary* library, const int* skill_ids, int count,
                          const char* name, const char* description) {
    if (!library || !skill_ids || count <= 0 || !name) return -1;

    SkillRecord composition;
    memset(&composition, 0, sizeof(SkillRecord));
    snprintf(composition.name, SKILL_MAX_NAME, "%s", name);
    snprintf(composition.description, SKILL_MAX_DESC, "%s", description ? description : "");
    composition.type = SKILL_TYPE_COMPOSITE;
    composition.enabled = 1;
    composition.created_at = time(NULL);

    composition.child_count = count < SKILL_MAX_PREREQ ? count : SKILL_MAX_PREREQ;
    for (int i = 0; i < composition.child_count; i++) {
        composition.child_skill_ids[i] = skill_ids[i];
    }

    for (int i = 0; i < composition.child_count; i++) {
        for (size_t j = 0; j < library->size; j++) {
            if (library->records[j].skill_id == skill_ids[i]) {
                for (int t = 0; t < library->records[j].tag_count && composition.tag_count < SKILL_MAX_TAGS; t++) {
                    int found = 0;
                    for (int et = 0; et < composition.tag_count; et++) {
                        if (strcmp(composition.tags[et], library->records[j].tags[t]) == 0) {
                            found = 1;
                            break;
                        }
                    }
                    if (!found) {
                        snprintf(composition.tags[composition.tag_count++], SKILL_MAX_TAG_LEN,
                                "%s", library->records[j].tags[t]);
                    }
                }
                if (composition.domain[0] == '\0' && library->records[j].domain[0] != '\0') {
                    snprintf(composition.domain, SKILL_MAX_DOMAIN, "%s", library->records[j].domain);
                }
                break;
            }
        }
    }

    generate_embedding_impl(&composition, composition.embedding, 64);
    return skill_library_add(library, &composition);
}

int skill_library_extract_from_execution(SkillLibrary* library, const char* execution_log) {
    if (!library || !execution_log) return -1;

    SkillRecord extracted;
    memset(&extracted, 0, sizeof(SkillRecord));
    snprintf(extracted.name, SKILL_MAX_NAME, "提取技能_%d", library->next_skill_id);
    snprintf(extracted.description, SKILL_MAX_DESC, "从执行日志自动提取");
    extracted.type = SKILL_TYPE_COMPOSITE;
    extracted.enabled = 1;
    extracted.created_at = time(NULL);

    const char* step_marker = execution_log;
    int step_count = 0;
    while ((step_marker = strstr(step_marker, "step:")) != NULL && step_count < SKILL_MAX_STEPS) {
        step_marker += 5;
        const char* step_end = strchr(step_marker, '\n');
        if (!step_end) step_end = step_marker + strlen(step_marker);
        size_t step_len = (size_t)(step_end - step_marker);
        if (step_len > SKILL_MAX_NAME - 1) step_len = SKILL_MAX_NAME - 1;
        snprintf(extracted.steps[step_count].action, SKILL_MAX_NAME, "%.*s", (int)step_len, step_marker);
        extracted.steps[step_count].estimated_time_ms = 1000.0f;
        extracted.steps[step_count].max_retries = 1;
        step_count++;
    }
    extracted.step_count = step_count;

    if (step_count == 0) return -1;

    snprintf(extracted.tags[0], SKILL_MAX_TAG_LEN, "自动提取");
    extracted.tag_count = 1;
    snprintf(extracted.domain, SKILL_MAX_DOMAIN, "自动学习");

    generate_embedding_impl(&extracted, extracted.embedding, 64);
    return skill_library_add(library, &extracted);
}

int skill_library_check_prerequisites(const SkillLibrary* library, int skill_id,
                                      const void* context) {
    if (!library || skill_id <= 0) return -1;

    SkillRecord record;
    if (skill_library_get(library, skill_id, &record) != 0) return -1;
    (void)context;
    if (record.prerequisite_count == 0) return 1;

    int satisfied = 0;
    for (int i = 0; i < record.prerequisite_count; i++) {
        if (record.prerequisites[i].type == SKILL_PREREQ_SKILL) {
            int pre_id = skill_library_find_by_name(library, record.prerequisites[i].value);
            if (pre_id > 0) satisfied++;
        } else {
            satisfied++;
        }
    }

    if (satisfied == record.prerequisite_count) return 1;
    if (satisfied > 0) return 0;
    return -1;
}

void skill_library_update_effectiveness(SkillLibrary* library, int skill_id,
                                        int success, float duration_ms, float quality) {
    if (!library || skill_id <= 0) return;
    for (size_t i = 0; i < library->size; i++) {
        if (library->records[i].skill_id == skill_id) {
            SkillEffectiveness* eff = &library->records[i].effectiveness;
            eff->execution_count++;
            if (success) eff->success_count++;
            float alpha = 0.1f;
            eff->success_rate = (1.0f - alpha) * eff->success_rate +
                               alpha * (float)eff->success_count / (float)eff->execution_count;
            eff->average_duration_ms = (1.0f - alpha) * eff->average_duration_ms + alpha * duration_ms;
            eff->quality_score = (1.0f - alpha) * eff->quality_score + alpha * quality;
            eff->last_executed = time(NULL);
            return;
        }
    }
}

int skill_library_get_stats(const SkillLibrary* library, SkillLibraryStats* stats) {
    if (!library || !stats) return -1;
    memcpy(stats, &library->stats, sizeof(SkillLibraryStats));
    return 0;
}

int skill_library_get_all_skills(const SkillLibrary* library, int* ids, int max_ids) {
    if (!library || !ids || max_ids <= 0) return 0;
    int count = 0;
    for (size_t i = 0; i < library->size && count < max_ids; i++) {
        if (library->records[i].enabled) {
            ids[count++] = library->records[i].skill_id;
        }
    }
    return count;
}

int skill_library_save(const SkillLibrary* library, const char* filepath) {
    if (!library || !filepath) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    uint32_t magic = 0x534B4C4C;
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&version, sizeof(uint32_t), 1, fp);
    fwrite(&library->size, sizeof(size_t), 1, fp);
    fwrite(&library->next_skill_id, sizeof(int), 1, fp);
    fwrite(library->records, sizeof(SkillRecord), library->size, fp);
    fclose(fp);
    return 0;
}

int skill_library_load(SkillLibrary* library, const char* filepath) {
    if (!library || !filepath) return -1;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    uint32_t magic, version;
    size_t saved_size;
    int saved_next_id;

    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 || magic != 0x534B4C4C) {
        fclose(fp);
        return -1;
    }
    fread(&version, sizeof(uint32_t), 1, fp);
    fread(&saved_size, sizeof(size_t), 1, fp);
    fread(&saved_next_id, sizeof(int), 1, fp);

    if (saved_size > library->capacity) {
        void* new_records = safe_realloc(library->records, saved_size * sizeof(SkillRecord));
        if (!new_records) {
            fclose(fp);
            return -1;
        }
        library->records = (SkillRecord*)new_records;
        library->capacity = saved_size;
    }

    library->size = saved_size;
    library->next_skill_id = saved_next_id;
    fread(library->records, sizeof(SkillRecord), saved_size, fp);
    fclose(fp);
    return 0;
}

int skill_library_get_dependencies(const SkillLibrary* library, int skill_id,
                                   int* dependency_ids, int max_deps) {
    if (!library || !dependency_ids || max_deps <= 0) return 0;
    SkillRecord record;
    if (skill_library_get(library, skill_id, &record) != 0) return 0;

    int count = 0;
    for (int i = 0; i < record.prerequisite_count && count < max_deps; i++) {
        if (record.prerequisites[i].type == SKILL_PREREQ_SKILL) {
            int pre_id = skill_library_find_by_name(library, record.prerequisites[i].value);
            if (pre_id > 0) {
                dependency_ids[count++] = pre_id;
            }
        }
    }
    return count;
}

int skill_library_get_next_skills(const SkillLibrary* library, int skill_id,
                                  int* results, int max_results) {
    if (!library || !results || max_results <= 0) return 0;
    SkillRecord current;
    if (skill_library_get(library, skill_id, &current) != 0) return 0;

    int count = 0;
    for (size_t i = 0; i < library->size && count < max_results; i++) {
        if (!library->records[i].enabled) continue;
        if (library->records[i].skill_id == skill_id) continue;

        for (int p = 0; p < library->records[i].prerequisite_count; p++) {
            if (library->records[i].prerequisites[p].type == SKILL_PREREQ_SKILL) {
                int pre_id = skill_library_find_by_name(library,
                    library->records[i].prerequisites[p].value);
                if (pre_id == skill_id) {
                    results[count++] = library->records[i].skill_id;
                    break;
                }
            }
        }
    }
    return count;
}
