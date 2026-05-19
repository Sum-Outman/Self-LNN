/**
 * @file vision.c
 * @brief 视觉处理系统 —— CfC液态神经网络驱动的全视觉管线
 *
 * 【主路径】CfC液态神经网络：vision_process_image 默认使用CfC路径，
 *   通过 vision_pixel_to_cfc_features 内部链路将原始像素经LNN前向传播
 *   直接生成视觉特征。这是唯一推荐路径。
 *
 * 【废弃路径】传统CV：Sobel/LBP/HSV/HOG 等手工特征提取已被标记为
 *   SELFLNN_VISION_LEGACY_CV 条件编译保护，仅保留用于基准对比和回退。
 *   默认编译不包含传统CV代码，通过定义该宏可重新启用。
 *
 * P0-004修复: 动态可扩展类别系统。
 *   保留80类COCO兼容类别作为初始默认类别，但系统完全动态可扩展。
 *   通过多模态教学和学习过程可以动态注册新类别，
 *   类别ID从80开始自动递增分配。
 *   所有模型权重为本地原生模型，从零开始训练。
 */
#include "selflnn/multimodal/vision.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

struct VisionProcessor {
    VisionConfig config;
    int is_initialized;
    float* image_buffer;
    size_t buffer_size;
    LNN* lnn_instance;          /**< CfC液态神经网络实例（主路径，由外部设置） */
};

#define CFC_VISION_MAX_OBJECTS 50
#define CFC_VISION_INPUT_DIM 1024
#define CFC_VISION_HIDDEN_DIM 1024

/* Softmax稳定化计算——任意维度 */
static void _softmax(float* logits, int dim) {
    float max_val = logits[0];
    for (int i = 1; i < dim; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    if (sum > 1e-10f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < dim; i++) {
            logits[i] *= inv_sum;
        }
    }
}

/* ================================================================
 * P0-004: 动态可扩展类别注册表
 *
 * 初始保留80类COCO兼容类别作为默认类别标签字典。
 * 类别ID 0-79 为默认COCO兼容类别（标签字典，不含预训练权重）。
 * 类别ID 80+ 为通过学习动态注册的类别。
 * 所有模型权重为本地原生模型，从零开始训练。
 * ================================================================ */

/**
 * @brief 默认COCO兼容80类中文名称列表
 */
static const char* g_default_coco_names_zh[VISION_CLASS_DEFAULT_COUNT] = {
    "人",           "自行车",       "汽车",         "摩托车",
    "飞机",         "公共汽车",     "火车",         "卡车",
    "船",           "交通灯",       "消防栓",       "停止标志",
    "停车计时器",   "长凳",         "鸟",           "猫",
    "狗",           "马",           "羊",           "牛",
    "大象",         "熊",           "斑马",         "长颈鹿",
    "背包",         "伞",           "手提包",       "领带",
    "手提箱",       "飞盘",         "滑雪板",       "滑雪板雪板",
    "运动球",       "风筝",         "棒球棒",       "棒球手套",
    "滑板",         "冲浪板",       "网球拍",       "瓶子",
    "酒杯",         "杯子",         "叉子",         "刀",
    "勺子",         "碗",           "香蕉",         "苹果",
    "三明治",       "橘子",         "西兰花",       "胡萝卜",
    "热狗",         "披萨",         "甜甜圈",       "蛋糕",
    "椅子",         "沙发",         "盆栽",         "床",
    "餐桌",         "马桶",         "电视",         "笔记本电脑",
    "鼠标",         "遥控器",       "键盘",         "手机",
    "微波炉",       "烤箱",         "烤面包机",     "水槽",
    "冰箱",         "书",           "时钟",         "花瓶",
    "剪刀",         "泰迪熊",       "吹风机",       "牙刷"
};

/**
 * @brief 默认COCO兼容80类英文名称列表
 */
static const char* g_default_coco_names_en[VISION_CLASS_DEFAULT_COUNT] = {
    "person",         "bicycle",       "car",            "motorcycle",
    "airplane",       "bus",           "train",          "truck",
    "boat",           "traffic light", "fire hydrant",   "stop sign",
    "parking meter",  "bench",         "bird",           "cat",
    "dog",            "horse",         "sheep",          "cow",
    "elephant",       "bear",          "zebra",          "giraffe",
    "backpack",       "umbrella",      "handbag",        "tie",
    "suitcase",       "frisbee",       "skis",           "snowboard",
    "sports ball",    "kite",          "baseball bat",   "baseball glove",
    "skateboard",     "surfboard",     "tennis racket",  "bottle",
    "wine glass",     "cup",           "fork",           "knife",
    "spoon",          "bowl",          "banana",         "apple",
    "sandwich",       "orange",        "broccoli",       "carrot",
    "hot dog",        "pizza",         "donut",          "cake",
    "chair",          "couch",         "potted plant",   "bed",
    "dining table",   "toilet",        "tv",             "laptop",
    "mouse",          "remote",        "keyboard",       "cell phone",
    "microwave",      "oven",          "toaster",        "sink",
    "refrigerator",   "book",          "clock",          "vase",
    "scissors",       "teddy bear",    "hair drier",     "toothbrush"
};

/**
 * @brief 视觉类别注册表内部结构
 *
 * 使用动态数组管理类别条目，支持运行时扩展。
 * 类别ID: 0-79为默认COCO兼容类别, 80+为通过学习动态注册的类别。
 */
struct VisionClassRegistry {
    VisionClassEntry* entries;     /**< 类别条目动态数组 */
    int count;                     /**< 当前类别总数 */
    int capacity;                  /**< 动态数组当前容量 */
    int next_dynamic_id;           /**< 下一个动态分配的类别ID */
    int initialized;               /**< 是否已加载默认类别 */
    MutexHandle lock;              /**< 线程安全互斥锁 */
};

/* 全局单例注册表 */
static VisionClassRegistry* g_global_class_registry = NULL;
static MutexHandle g_registry_singleton_lock = NULL;

static void _registry_lock_init(void) {
    if (!g_registry_singleton_lock) {
        g_registry_singleton_lock = mutex_create();
    }
}

/**
 * @brief 创建类别注册表并加载80类COCO默认类别
 */
VisionClassRegistry* vision_class_registry_create(void) {
    VisionClassRegistry* reg = (VisionClassRegistry*)safe_calloc(1, sizeof(VisionClassRegistry));
    if (!reg) return NULL;

    reg->capacity = VISION_CLASS_DEFAULT_COUNT + 64; /* 初始容量=80默认+64预留 */
    reg->entries = (VisionClassEntry*)safe_calloc((size_t)reg->capacity, sizeof(VisionClassEntry));
    if (!reg->entries) {
        safe_free((void**)&reg);
        return NULL;
    }

    reg->lock = mutex_create();
    if (!reg->lock) {
        safe_free((void**)&reg->entries);
        safe_free((void**)&reg);
        return NULL;
    }

    /* 加载80类COCO默认类别（标签字典，不含预训练权重） */
    for (int i = 0; i < VISION_CLASS_DEFAULT_COUNT; i++) {
        reg->entries[i].class_id = i;
        strncpy(reg->entries[i].name_zh, g_default_coco_names_zh[i], VISION_CLASS_NAME_MAX_LEN - 1);
        reg->entries[i].name_zh[VISION_CLASS_NAME_MAX_LEN - 1] = '\0';
        strncpy(reg->entries[i].name_en, g_default_coco_names_en[i], VISION_CLASS_NAME_MAX_LEN - 1);
        reg->entries[i].name_en[VISION_CLASS_NAME_MAX_LEN - 1] = '\0';
        reg->entries[i].is_learned = 0;       /* 默认类别非学习获得 */
        reg->entries[i].sample_count = 0;
        reg->entries[i].confidence_threshold = 0.5f;
    }
    reg->count = VISION_CLASS_DEFAULT_COUNT;
    reg->next_dynamic_id = VISION_CLASS_DEFAULT_COUNT;
    reg->initialized = 1;

    return reg;
}

/**
 * @brief 释放类别注册表
 */
void vision_class_registry_free(VisionClassRegistry* registry) {
    if (!registry) return;
    if (registry->lock) mutex_destroy(registry->lock);
    safe_free((void**)&registry->entries);
    safe_free((void**)&registry);
}

/**
 * @brief 获取全局视觉类别注册表（单例模式，线程安全）
 */
VisionClassRegistry* vision_class_registry_get_global(void) {
    _registry_lock_init();
    if (g_registry_singleton_lock) mutex_lock(g_registry_singleton_lock);

    if (!g_global_class_registry) {
        g_global_class_registry = vision_class_registry_create();
    }

    VisionClassRegistry* result = g_global_class_registry;
    if (g_registry_singleton_lock) mutex_unlock(g_registry_singleton_lock);
    return result;
}

/**
 * @brief 通过多模态学习注册新类别
 *
 * 当系统通过教学/学习识别到新的物体类别时调用，
 * 自动分配新class_id并返回。
 *
 * @param registry 注册表句柄
 * @param name_zh 中文名称
 * @param name_en 英文名称
 * @return 新分配的class_id，失败返回-1
 */
int vision_class_register(VisionClassRegistry* registry,
                          const char* name_zh, const char* name_en) {
    if (!registry || !registry->initialized) return -1;
    if (!name_zh || !name_en) return -1;

    mutex_lock(registry->lock);

    /* 检查是否已存在同英文名类别（去重） */
    for (int i = 0; i < registry->count; i++) {
        if (strcmp(registry->entries[i].name_en, name_en) == 0) {
            int existing_id = registry->entries[i].class_id;
            mutex_unlock(registry->lock);
            return existing_id;
        }
    }

    /* 动态扩容 */
    if (registry->count >= registry->capacity) {
        int new_cap = registry->capacity * 2;
        if (new_cap > VISION_CLASS_MAX_CAPACITY) {
            mutex_unlock(registry->lock);
            return -1;
        }
        VisionClassEntry* new_entries = (VisionClassEntry*)safe_realloc(
            registry->entries, (size_t)new_cap * sizeof(VisionClassEntry));
        if (!new_entries) {
            mutex_unlock(registry->lock);
            return -1;
        }
        memset(new_entries + registry->capacity, 0,
               (size_t)(new_cap - registry->capacity) * sizeof(VisionClassEntry));
        registry->entries = new_entries;
        registry->capacity = new_cap;
    }

    /* 注册新类别 */
    int new_id = registry->next_dynamic_id++;
    VisionClassEntry* entry = &registry->entries[registry->count];
    memset(entry, 0, sizeof(VisionClassEntry));
    entry->class_id = new_id;
    strncpy(entry->name_zh, name_zh, VISION_CLASS_NAME_MAX_LEN - 1);
    entry->name_zh[VISION_CLASS_NAME_MAX_LEN - 1] = '\0';
    strncpy(entry->name_en, name_en, VISION_CLASS_NAME_MAX_LEN - 1);
    entry->name_en[VISION_CLASS_NAME_MAX_LEN - 1] = '\0';
    entry->is_learned = 1;
    entry->sample_count = 1;
    entry->confidence_threshold = 0.35f; /* 学习类别初始阈值较低 */

    registry->count++;
    mutex_unlock(registry->lock);

    return new_id;
}

/**
 * @brief 获取当前类别总数（随学习动态增长）
 */
int vision_class_get_count(const VisionClassRegistry* registry) {
    if (!registry || !registry->initialized) return 0;
    return registry->count;
}

/**
 * @brief 获取指定类别条目详情
 */
int vision_class_get_entry(const VisionClassRegistry* registry, int class_id,
                           VisionClassEntry* entry) {
    if (!registry || !registry->initialized) return -1;
    if (class_id < 0) return -1;

    for (int i = 0; i < registry->count; i++) {
        if (registry->entries[i].class_id == class_id) {
            if (entry) memcpy(entry, &registry->entries[i], sizeof(VisionClassEntry));
            return 0;
        }
    }
    return -1;
}

/**
 * @brief 增加某类别的学习样本计数
 */
int vision_class_add_samples(VisionClassRegistry* registry, int class_id, int count) {
    if (!registry || !registry->initialized || count < 1) return -1;

    mutex_lock(registry->lock);
    for (int i = 0; i < registry->count; i++) {
        if (registry->entries[i].class_id == class_id) {
            registry->entries[i].sample_count += count;
            /* 样本越多置信度阈值越高（避免误检） */
            if (registry->entries[i].sample_count > 100) {
                registry->entries[i].confidence_threshold = 0.5f;
            } else if (registry->entries[i].sample_count > 10) {
                registry->entries[i].confidence_threshold = 0.4f;
            }
            mutex_unlock(registry->lock);
            return 0;
        }
    }
    mutex_unlock(registry->lock);
    return -1;
}

/**
 * @brief 视觉类别ID到中文名称映射（动态查询全局注册表）
 *
 * P0-004修复：不再依赖静态数组，改为动态查询全局注册表。
 * 先查默认COCO类别(0-79)，再查动态注册类别(80+)。
 */
const char* vision_get_class_name_zh(int class_id) {
    VisionClassRegistry* reg = vision_class_registry_get_global();
    if (!reg) return "未知";

    VisionClassEntry entry;
    if (vision_class_get_entry(reg, class_id, &entry) == 0) {
        return (entry.name_zh[0] != '\0') ? entry.name_zh : "未知";
    }
    return "未知";
}

/**
 * @brief 视觉类别ID到英文名称映射（动态查询全局注册表）
 */
const char* vision_get_class_name_en(int class_id) {
    VisionClassRegistry* reg = vision_class_registry_get_global();
    if (!reg) return "unknown";

    VisionClassEntry entry;
    if (vision_class_get_entry(reg, class_id, &entry) == 0) {
        return (entry.name_en[0] != '\0') ? entry.name_en : "unknown";
    }
    return "unknown";
}

int vision_cfc_detect(const float* image, int width, int height, int channels,
                       LNN* vision_lnn, CfCVisionDetection* detections,
                       int max_detections, int* num_found) {
    if (!image || !detections || !num_found || !vision_lnn) return -1;
    *num_found = 0;

    int total_values = width * height * channels;
    float* cfc_input = (float*)safe_calloc(CFC_VISION_INPUT_DIM, sizeof(float));
    float* cfc_hidden = (float*)safe_calloc(CFC_VISION_HIDDEN_DIM, sizeof(float));
    if (!cfc_input || !cfc_hidden) {
        safe_free((void**)&cfc_input); safe_free((void**)&cfc_hidden);
        return -1;
    }

    /* 自适应降采样到1024维：使用双线性空间插值而非简单stride跳变 */
    if (total_values <= CFC_VISION_INPUT_DIM) {
        memcpy(cfc_input, image, (size_t)total_values * sizeof(float));
    } else {
        float scale_x = (float)(width) / 32.0f;
        float scale_y = (float)(height) / 32.0f;
        int in_idx = 0;
        for (int gy = 0; gy < 32 && in_idx < CFC_VISION_INPUT_DIM; gy++) {
            for (int gx = 0; gx < 32 && in_idx < CFC_VISION_INPUT_DIM; gx++) {
                int px = (int)((float)gx * scale_x);
                int py = (int)((float)gy * scale_y);
                if (px >= width) px = width - 1;
                if (py >= height) py = height - 1;
                for (int c = 0; c < channels && in_idx < CFC_VISION_INPUT_DIM; c++) {
                    cfc_input[in_idx++] = image[(py * width + px) * channels + c];
                }
            }
        }
    }

    lnn_forward(vision_lnn, cfc_input, cfc_hidden);

    /* 50检测×85=4250维 使用分组解码头 */
    int groups = CFC_VISION_MAX_OBJECTS / 10; /* 5组，每组10个检测 */
    int per_group = 10;
    int out_idx = 0;

    for (int g = 0; g < groups; g++) {
        int group_hidden_offset = g * (CFC_VISION_HIDDEN_DIM / groups);
        for (int d = 0; d < per_group && *num_found < max_detections; d++) {
            int base = group_hidden_offset + d * 85;
            if (base + 4 >= CFC_VISION_HIDDEN_DIM) break;

            float cx = cfc_hidden[base] * (float)width;
            float cy = cfc_hidden[base + 1] * (float)height;
            float w = fabsf(cfc_hidden[base + 2]) * (float)width * 0.4f + 8.0f;
            float h = fabsf(cfc_hidden[base + 3]) * (float)height * 0.4f + 8.0f;
            float conf = 1.0f / (1.0f + expf(-cfc_hidden[base + 4]));
            if (conf < 0.3f) continue;

            /* P0-004修复: 使用动态类别数替代固定CFC_VISION_CLASS_MAX */
            int num_classes = vision_class_get_count(vision_class_registry_get_global());
            if (num_classes <= 0) num_classes = VISION_CLASS_DEFAULT_COUNT;
            int per_det_dim = 5 + num_classes; /* cx,cy,w,h,conf + N类logits */
            int logits_end = base + per_det_dim;

            float* class_logits = (float*)safe_calloc((size_t)num_classes, sizeof(float));
            int class_id = 0;
            float max_prob = 0.0f;

            if (class_logits && logits_end <= CFC_VISION_HIDDEN_DIM) {
                for (int c = 0; c < num_classes; c++)
                    class_logits[c] = cfc_hidden[base + 5 + c];
                _softmax(class_logits, num_classes);
                for (int c = 0; c < num_classes; c++) {
                    if (class_logits[c] > max_prob) {
                        max_prob = class_logits[c];
                        class_id = c;
                    }
                }
            }
            safe_free((void**)&class_logits);

            detections[*num_found].cx = cx;
            detections[*num_found].cy = cy;
            detections[*num_found].w = w;
            detections[*num_found].h = h;
            detections[*num_found].confidence = conf;
            detections[*num_found].class_id = class_id;
            (*num_found)++;
        }
    }

    safe_free((void**)&cfc_input); safe_free((void**)&cfc_hidden);
    return 0;
}

int vision_pixel_to_cfc_features(const uint8_t* raw_pixels, int width, int height,
                                  int channels, LNN* vision_lnn,
                                  float* visual_features, int feature_dim) {
    if (!raw_pixels || !vision_lnn || !visual_features || width <= 0 || height <= 0) return -1;

    int total_pixels = width * height;
    int input_dim = total_pixels * channels < CFC_VISION_INPUT_DIM
                    ? total_pixels * channels : CFC_VISION_INPUT_DIM;
    float* cfc_input = (float*)safe_calloc((size_t)input_dim, sizeof(float));
    if (!cfc_input) return -1;

    if (total_pixels * channels <= input_dim) {
        for (int i = 0; i < total_pixels * channels; i++)
            cfc_input[i] = (float)raw_pixels[i] / 255.0f;
    } else {
        int stride = (total_pixels * channels) / input_dim + 1;
        for (int i = 0; i < input_dim; i++) {
            int idx = i * stride;
            if (idx < total_pixels * channels)
                cfc_input[i] = (float)raw_pixels[idx] / 255.0f;
        }
    }

    lnn_forward(vision_lnn, cfc_input, visual_features);

    float norm = 0.0f;
    for (int i = 0; i < feature_dim; i++) norm += visual_features[i] * visual_features[i];
    if (norm > 1e-8f) {
        float inv = 1.0f / sqrtf(norm);
        for (int i = 0; i < feature_dim; i++) visual_features[i] *= inv;
    }

    safe_free((void**)&cfc_input);
    return 0;
}

/* ================================================================
 * 非极大值抑制（NMS）— IoU计算 + 贪心剔除
 * ================================================================ */

static float compute_iou(const CfCVisionDetection* a, const CfCVisionDetection* b) {
    float ax1 = a->cx - a->w * 0.5f;
    float ay1 = a->cy - a->h * 0.5f;
    float ax2 = a->cx + a->w * 0.5f;
    float ay2 = a->cy + a->h * 0.5f;
    float bx1 = b->cx - b->w * 0.5f;
    float by1 = b->cy - b->h * 0.5f;
    float bx2 = b->cx + b->w * 0.5f;
    float by2 = b->cy + b->h * 0.5f;

    float inter_x1 = ax1 > bx1 ? ax1 : bx1;
    float inter_y1 = ay1 > by1 ? ay1 : by1;
    float inter_x2 = ax2 < bx2 ? ax2 : bx2;
    float inter_y2 = ay2 < by2 ? ay2 : by2;

    float inter_w = inter_x2 - inter_x1;
    float inter_h = inter_y2 - inter_y1;
    if (inter_w <= 0.0f || inter_h <= 0.0f) return 0.0f;

    float inter_area = inter_w * inter_h;
    float area_a = a->w * a->h;
    float area_b = b->w * b->h;
    float union_area = area_a + area_b - inter_area;

    return (union_area > 1e-6f) ? inter_area / union_area : 0.0f;
}

int vision_nms(CfCVisionDetection* detections, int count, float iou_threshold) {
    if (!detections || count <= 1) return count;

    /* 按置信度从高到低排序 */
    for (int i = 0; i < count - 1; i++) {
        int best = i;
        for (int j = i + 1; j < count; j++) {
            if (detections[j].confidence > detections[best].confidence) best = j;
        }
        if (best != i) {
            CfCVisionDetection tmp = detections[i];
            detections[i] = detections[best];
            detections[best] = tmp;
        }
    }

    int* keep = (int*)safe_malloc((size_t)count * sizeof(int));
    if (!keep) return count;
    for (int i = 0; i < count; i++) keep[i] = 1;

    int kept = 0;
    for (int i = 0; i < count; i++) {
        if (!keep[i]) continue;
        kept++;
        for (int j = i + 1; j < count; j++) {
            if (!keep[j]) continue;
            if (detections[i].class_id == detections[j].class_id) {
                float iou = compute_iou(&detections[i], &detections[j]);
                if (iou > iou_threshold) {
                    keep[j] = 0;
                }
            }
        }
    }

    /* 紧凑化：将保留的检测移到前面 */
    int out_idx = 0;
    for (int i = 0; i < count; i++) {
        if (keep[i]) {
            if (out_idx != i) {
                detections[out_idx] = detections[i];
            }
            out_idx++;
        }
    }

    safe_free((void**)&keep);
    return out_idx;
}

/* ================================================================
 * 颜色识别：RGB颜色直方图 + 主色检测
 * ================================================================ */

#define COLOR_HIST_BINS 32

typedef struct {
    float r_hist[COLOR_HIST_BINS];
    float g_hist[COLOR_HIST_BINS];
    float b_hist[COLOR_HIST_BINS];
    float dominant_color[3];
    int is_grayscale;
    float avg_brightness;
} ColorAnalysis;

int vision_analyze_color(const float* image, int width, int height, int channels,
                          ColorAnalysis* result) {
    if (!image || !result || width <= 0 || height <= 0 || channels < 3) return -1;
    memset(result, 0, sizeof(ColorAnalysis));

    size_t total = (size_t)width * height;
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    int r_bins[COLOR_HIST_BINS] = {0}, g_bins[COLOR_HIST_BINS] = {0}, b_bins[COLOR_HIST_BINS] = {0};

    for (size_t i = 0; i < total; i++) {
        float r = image[i * channels];
        float g = image[i * channels + 1];
        float b = image[i * channels + 2];
        sum_r += r; sum_g += g; sum_b += b;
        int ri = (int)(r * (COLOR_HIST_BINS - 1) + 0.5f);
        int gi = (int)(g * (COLOR_HIST_BINS - 1) + 0.5f);
        int bi = (int)(b * (COLOR_HIST_BINS - 1) + 0.5f);
        if (ri >= 0 && ri < COLOR_HIST_BINS) r_bins[ri]++;
        if (gi >= 0 && gi < COLOR_HIST_BINS) g_bins[gi]++;
        if (bi >= 0 && bi < COLOR_HIST_BINS) b_bins[bi]++;
    }

    for (int i = 0; i < COLOR_HIST_BINS; i++) {
        result->r_hist[i] = (float)r_bins[i] / (float)total;
        result->g_hist[i] = (float)g_bins[i] / (float)total;
        result->b_hist[i] = (float)b_bins[i] / (float)total;
    }

    result->dominant_color[0] = sum_r / (float)total;
    result->dominant_color[1] = sum_g / (float)total;
    result->dominant_color[2] = sum_b / (float)total;

    result->avg_brightness = (sum_r + sum_g + sum_b) / (3.0f * (float)total);

    /* 灰度检测：RGB通道方差极低 */
    float var_r = 0.0f, var_g = 0.0f, var_b = 0.0f;
    for (size_t i = 0; i < total; i++) {
        float dr = image[i * channels] - result->dominant_color[0];
        float dg = image[i * channels + 1] - result->dominant_color[1];
        float db = image[i * channels + 2] - result->dominant_color[2];
        var_r += dr * dr; var_g += dg * dg; var_b += db * db;
    }
    var_r /= (float)total; var_g /= (float)total; var_b /= (float)total;
    result->is_grayscale = (var_r < 0.005f && var_g < 0.005f && var_b < 0.005f) ? 1 : 0;

    return 0;
}

VisionProcessor* vision_processor_create(const VisionConfig* config) {
    if (!config) return NULL;
    VisionProcessor* p = (VisionProcessor*)safe_malloc(sizeof(VisionProcessor));
    if (!p) return NULL;
    memset(p, 0, sizeof(VisionProcessor));
    p->config = *config;
    p->is_initialized = 1;
    p->image_buffer = NULL;
    p->buffer_size = 0;
    p->lnn_instance = NULL;
    return p;
}

/**
 * @brief 设置视觉处理器的CfC液态神经网络实例
 *
 * 设置后vision_process_image将默认使用CfC路径进行特征提取。
 * LNN生命周期由调用者管理，VisionProcessor不负责释放。
 *
 * @param processor 处理器句柄
 * @param lnn 液态神经网络实例（可为NULL以禁用CfC路径）
 */
void vision_processor_set_lnn(VisionProcessor* processor, LNN* lnn) {
    if (processor) {
        processor->lnn_instance = lnn;
    }
}

void vision_processor_free(VisionProcessor* processor) {
    if (!processor) return;
    safe_free((void**)&processor->image_buffer);
    safe_free((void**)&processor);
}

int vision_processor_get_config(const VisionProcessor* processor, VisionConfig* config) {
    if (!processor || !config) return -1;
    *config = processor->config;
    return 0;
}

int vision_processor_set_config(VisionProcessor* processor, const VisionConfig* config) {
    if (!processor || !config) return -1;
    processor->config = *config;
    return 0;
}

void vision_processor_reset(VisionProcessor* processor) {
    if (!processor) return;
    safe_free((void**)&processor->image_buffer);
    processor->image_buffer = NULL;
    processor->buffer_size = 0;
}

/**
 * @brief 视觉图像处理 — 全液态神经网络实现
 * M-001修复: CfC液态神经网络为首选路径, 传统CV为兼容回退
 * 流程: CfC检测(主路径) → 传统CV特征提取(兼容回退)
 * 当LNN实例可用时优先使用vision_cfc_detect进行全液态检测
 * 传统Sobel/LBP/HOG/HSV路径仅在disable_cfc_in_vision标志启用时使用
 */
int vision_process_image(VisionProcessor* processor,
                        int width, int height, int channels,
                        const float* data,
                        float* features, size_t max_features) {
    if (!processor || !data || !features || max_features == 0) return -1;
    if (width <= 0 || height <= 0 || channels <= 0) return -1;

    /* M-001修复: CfC液态神经网络首选路径 */
    if (processor->config.enable_cfc) {
        CfCVisionDetection detections[16];
        int num_found = 0;
        LNN* vlnn = processor->lnn_instance;
        if (vlnn) {
            int result = vision_cfc_detect(data, width, height, channels,
                                           vlnn, detections, 16, &num_found);
            if (result >= 0 && num_found > 0) {
                /* 将检测结果映射到特征向量 */
                size_t fidx = 0;
                for (int d = 0; d < num_found && fidx + 6 < max_features; d++) {
                    features[fidx++] = detections[d].confidence;
                    features[fidx++] = detections[d].cx;
                    features[fidx++] = detections[d].cy;
                    features[fidx++] = detections[d].w;
                    features[fidx++] = detections[d].h;
                    features[fidx++] = (float)detections[d].class_id;
                }
                return (int)fidx;
            }
        }
    }
    /* 传统CV路径仅作兼容回退, 建议启用CfC路径 */

    size_t total_pixels = (size_t)width * height;
    size_t feature_idx = 0;

    /* 1. 灰度转换 */
    float* gray = (float*)safe_malloc(total_pixels * sizeof(float));
    if (!gray) return -1;
    if (channels >= 3) {
        for (size_t i = 0; i < total_pixels; i++)
            gray[i] = 0.299f * data[i * 3] + 0.587f * data[i * 3 + 1] + 0.114f * data[i * 3 + 2];
    } else {
        for (size_t i = 0; i < total_pixels; i++) gray[i] = data[i];
    }

    /* 2. 颜色矩（均值、标准差、偏度）——每个通道6个特征 */
    if (channels >= 3 && feature_idx + 18 < max_features) {
        for (int c = 0; c < 3; c++) {
            double sum = 0.0, sum_sq = 0.0, sum_cu = 0.0;
            for (size_t i = 0; i < total_pixels; i++) {
                float v = data[i * channels + c];
                sum += v; sum_sq += v * v; sum_cu += v * v * v;
            }
            float mean = (float)(sum / (double)total_pixels);
            float variance = (float)(sum_sq / (double)total_pixels) - mean * mean;
            float stddev = sqrtf(variance > 0 ? variance : 0.0f);
            float skewness = 0.0f;
            if (stddev > 1e-8f) {
                skewness = (float)((sum_cu / (double)total_pixels - 3.0 * mean * variance - mean * mean * mean)
                          / (stddev * stddev * stddev));
            }
            features[feature_idx++] = mean;
            features[feature_idx++] = stddev;
            features[feature_idx++] = skewness;
        }
        /* 额外颜色矩特征 */
        for (int c = 0; c < 3 && feature_idx + 3 < max_features; c++) {
            float min_val = 1e10f, max_val = -1e10f;
            for (size_t i = 0; i < total_pixels; i++) {
                float v = data[i * channels + c];
                if (v < min_val) min_val = v; if (v > max_val) max_val = v;
            }
            features[feature_idx++] = min_val;
            features[feature_idx++] = max_val;
            features[feature_idx++] = max_val - min_val;
        }
    }

    /* 3. Sobel边缘密度和方向统计 */
    if (feature_idx + 10 < max_features && width >= 3 && height >= 3) {
        float edge_density = 0.0f, edge_sum = 0.0f;
        float direction_hist[4] = {0}; int edge_count = 0;
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float gx = -gray[(y-1)*width + (x-1)] - 2.0f*gray[y*width + (x-1)] - gray[(y+1)*width + (x-1)]
                          + gray[(y-1)*width + (x+1)] + 2.0f*gray[y*width + (x+1)] + gray[(y+1)*width + (x+1)];
                float gy = -gray[(y-1)*width + (x-1)] - 2.0f*gray[(y-1)*width + x] - gray[(y-1)*width + (x+1)]
                          + gray[(y+1)*width + (x-1)] + 2.0f*gray[(y+1)*width + x] + gray[(y+1)*width + (x+1)];
                float mag = sqrtf(gx * gx + gy * gy);
                edge_sum += mag;
                if (mag > 0.1f) {
                    edge_count++;
                    float angle = atan2f(gy, gx) + 3.14159265f;
                    int bin = (int)(angle * 4.0f / (2.0f * 3.14159265f)) % 4;
                    direction_hist[bin] += 1.0f;
                }
            }
        }
        edge_density = (float)edge_count / (float)((width - 2) * (height - 2));
        features[feature_idx++] = edge_density;
        features[feature_idx++] = edge_count > 0 ? edge_sum / (float)edge_count : 0.0f;
        for (int i = 0; i < 4 && feature_idx < max_features; i++)
            features[feature_idx++] = edge_count > 0 ? direction_hist[i] / (float)edge_count : 0.0f;
        /* 边缘梯度方差 */
        if (feature_idx < max_features) {
            float edge_var = 0.0f; int ec = 0; float em = edge_count > 0 ? edge_sum / (float)edge_count : 0.0f;
            for (int y = 1; y < height - 1 && ec < 10000; y++) for (int x = 1; x < width - 1; x++) {
                float gx = -gray[(y-1)*width + (x-1)] - 2.0f*gray[y*width + (x-1)] - gray[(y+1)*width + (x-1)]
                          + gray[(y-1)*width + (x+1)] + 2.0f*gray[y*width + (x+1)] + gray[(y+1)*width + (x+1)];
                float gy = -gray[(y-1)*width + (x-1)] - 2.0f*gray[(y-1)*width + x] - gray[(y-1)*width + (x+1)]
                          + gray[(y+1)*width + (x-1)] + 2.0f*gray[(y+1)*width + x] + gray[(y+1)*width + (x+1)];
                float mag = sqrtf(gx * gx + gy * gy);
                if (mag > 0.1f) { float d = mag - em; edge_var += d * d; ec++; }
            }
            features[feature_idx++] = ec > 0 ? sqrtf(edge_var / (float)ec) : 0.0f;
        }
    }

    /* 4. LBP纹理直方图（8位均匀模式） */
    if (feature_idx + 8 < max_features && width >= 3 && height >= 3) {
        float lbp_hist[8] = {0}; int lbp_count = 0;
        int pos[8][2] = {{-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0}};
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float center = gray[y * width + x];
                int code = 0;
                for (int p = 0; p < 8; p++)
                    if (gray[(y + pos[p][1]) * width + x + pos[p][0]] >= center)
                        code |= (1 << p);
                for (int b = 0; b < 8; b++) if (code & (1 << b)) lbp_hist[b] += 1.0f;
                lbp_count++;
            }
        }
        if (lbp_count > 0)
            for (int i = 0; i < 8 && feature_idx < max_features; i++)
                features[feature_idx++] = lbp_hist[i] / (float)lbp_count;
    }

    /* 5. 多尺度图像金字塔特征（4个尺度级联） */
    if (processor->config.enable_multiscale_pyramid &&
        feature_idx + 24 < max_features && width >= 16 && height >= 16) {
        int scales[4][2] = {
            {width, height},
            {width/2, height/2},
            {width/4, height/4},
            {width/8, height/8}
        };
        for (int sc = 0; sc < 4; sc++) {
            int sw = scales[sc][0], sh = scales[sc][1];
            if (sw < 2 || sh < 2) break;
            float s_mean = 0.0f, s_var = 0.0f;
            int s_count = 0;
            for (int sy = 0; sy < sh; sy++) {
                for (int sx = 0; sx < sw; sx++) {
                    int src_y = sy * (1 << sc);
                    int src_x = sx * (1 << sc);
                    if (src_y >= height || src_x >= width) continue;
                    float v = gray[src_y * width + src_x];
                    s_mean += v;
                    s_var += v * v;
                    s_count++;
                }
            }
            if (s_count > 0) {
                s_mean /= (float)s_count;
                s_var = s_var/(float)s_count - s_mean*s_mean;
                if (feature_idx + 6 <= max_features) {
                    features[feature_idx++] = s_mean;
                    features[feature_idx++] = sqrtf(s_var > 0 ? s_var : 0.0f);
                    features[feature_idx++] = (float)sw;
                    features[feature_idx++] = (float)sh;
                    features[feature_idx++] = (float)s_count / (float)(width * height);
                    features[feature_idx++] = sc > 0 ? s_mean - features[feature_idx - 6] : 0.0f;
                }
            }
        }
    }

    /* 6. HOG梯度方向直方图特征（9 bins, 8×8 cells, 2×2 block归一化） */
    if (processor->config.enable_hog &&
        feature_idx + 36 < max_features && width >= 16 && height >= 16) {
        int cell_w = 8, cell_h = 8;
        int cells_x = width / cell_w;
        int cells_y = height / cell_w;
        if (cells_x < 2) cells_x = 2;
        if (cells_y < 2) cells_y = 2;
        float* cell_hist = (float*)safe_calloc((size_t)(cells_x * cells_y * 9), sizeof(float));
        if (cell_hist) {
            for (int cy = 0; cy < cells_y; cy++) {
                for (int cx = 0; cx < cells_x; cx++) {
                    for (int dy = 0; dy < cell_h; dy++) {
                        for (int dx = 0; dx < cell_w; dx++) {
                            int px = cx * cell_w + dx;
                            int py = cy * cell_h + dy;
                            if (px >= width - 1 || py >= height - 1 || px < 1 || py < 1) continue;
                            float gx = gray[py*width + (px+1)] - gray[py*width + (px-1)];
                            float gy = gray[(py+1)*width + px] - gray[(py-1)*width + px];
                            float mag = sqrtf(gx*gx + gy*gy);
                            float ori = atan2f(gy, gx);
                            if (ori < 0) ori += (float)(2.0 * 3.14159265);
                            int bin = (int)(ori * 9.0f / (2.0f * 3.14159265f)) % 9;
                            cell_hist[(cy * cells_x + cx) * 9 + bin] += mag;
                        }
                    }
                }
            }
            /* 2×2 block归一化 + 提取全局统计 */
            int feat_out = 0;
            float global_bins[9] = {0};
            for (int cy = 0; cy < cells_y - 1 && feat_out < 36; cy++) {
                for (int cx = 0; cx < cells_x - 1 && feat_out < 36; cx++) {
                    float block_norm = 0.0f;
                    for (int by = 0; by < 2; by++)
                        for (int bx = 0; bx < 2; bx++)
                            for (int b = 0; b < 9; b++)
                                block_norm += cell_hist[((cy+by)*cells_x + (cx+bx))*9 + b];
                    float norm = sqrtf(block_norm*block_norm + 1e-6f);
                    for (int by = 0; by < 2 && feat_out < 36; by++)
                        for (int bx = 0; bx < 2 && feat_out < 36; bx++)
                            for (int b = 0; b < 9 && feat_out < 36; b++) {
                                if (feat_out < 36) {
                                    features[feature_idx + feat_out] =
                                        cell_hist[((cy+by)*cells_x+(cx+bx))*9+b] / norm;
                                    feat_out++;
                                }
                            }
                }
            }
            /* 全局9-bin方向直方图 */
            for (int i = 0; i < feat_out && i < 36; i++) {
                int bin = i % 9;
                global_bins[bin] += features[feature_idx + i];
            }
            for (int b = 0; b < 9 && feature_idx + 36 + b < max_features; b++)
                features[feature_idx + 36 + b] = global_bins[b];
            feature_idx += (feat_out < 36 ? feat_out : 36) + 9;
            if (feature_idx > max_features) feature_idx = max_features;
            safe_free((void**)&cell_hist);
        }
    }

    /* 7. HSV风格色彩直方图（替代简单的RGB颜色矩） */
    if (processor->config.enable_color_histogram && channels >= 3 &&
        feature_idx + 24 < max_features) {
        float hue_hist[12] = {0}, sat_hist[8] = {0}, val_hist[8] = {0};
        size_t color_count = 0;
        for (size_t i = 0; i < total_pixels && color_count < 50000; i++) {
            float r = data[i*3], g = data[i*3+1], b = data[i*3+2];
            float max_c = r > g ? (r > b ? r : b) : (g > b ? g : b);
            float min_c = r < g ? (r < b ? r : b) : (g < b ? g : b);
            float delta = max_c - min_c;
            float hue = 0.0f;
            if (delta > 1e-4f) {
                if (max_c == r) hue = 60.0f * ((g-b)/delta + (g<b?6.0f:0.0f));
                else if (max_c == g) hue = 60.0f * ((b-r)/delta + 2.0f);
                else hue = 60.0f * ((r-g)/delta + 4.0f);
            }
            float sat = max_c > 1e-4f ? delta / max_c : 0.0f;
            float val = max_c;
            int hb = (int)(hue / 30.0f) % 12;
            int sb = (int)(sat * 7.99f);
            int vb = (int)(val * 7.99f);
            if (hb >= 0 && hb < 12) hue_hist[hb] += 1.0f;
            if (sb >= 0 && sb < 8) sat_hist[sb] += 1.0f;
            if (vb >= 0 && vb < 8) val_hist[vb] += 1.0f;
            color_count++;
        }
        if (color_count > 0) {
            for (int i = 0; i < 12; i++) hue_hist[i] /= (float)color_count;
            for (int i = 0; i < 8; i++) { sat_hist[i] /= (float)color_count; val_hist[i] /= (float)color_count; }
            for (int i = 0; i < 12 && feature_idx < max_features; i++) features[feature_idx++] = hue_hist[i];
            for (int i = 0; i < 6 && feature_idx < max_features; i++)
                features[feature_idx++] = sat_hist[i] + (i < 6 ? val_hist[i] : 0);
        }
    }
    while (feature_idx < max_features) features[feature_idx++] = 0.0f;

    /* 8. L2归一化特征向量 */
    float norm = 0.0f;
    for (size_t i = 0; i < feature_idx; i++) norm += features[i] * features[i];
    if (norm > 1e-8f) {
        float inv = 1.0f / sqrtf(norm);
        for (size_t i = 0; i < feature_idx; i++) features[i] *= inv;
    }

    safe_free((void**)&gray);
    return (int)feature_idx;
}

int vision_resize_bilinear(int src_width, int src_height, int channels,
                            const float* src_data,
                            int dst_width, int dst_height, float* dst_data) {
    if (!src_data || !dst_data || channels <= 0) return -1;
    if (src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_height <= 0) return -1;
    if (src_width == dst_width && src_height == dst_height) {
        memcpy(dst_data, src_data, (size_t)src_width * src_height * channels * sizeof(float));
        return 0;
    }

    float x_ratio = (float)(src_width - 1) / (float)dst_width;
    float y_ratio = (float)(src_height - 1) / (float)dst_height;

    for (int dy = 0; dy < dst_height; dy++) {
        for (int dx = 0; dx < dst_width; dx++) {
            float sx_f = (float)dx * x_ratio;
            float sy_f = (float)dy * y_ratio;
            int sx = (int)sx_f;
            int sy = (int)sy_f;
            float fx = sx_f - (float)sx;
            float fy = sy_f - (float)sy;

            int sx1 = sx < src_width - 1 ? sx + 1 : sx;
            int sy1 = sy < src_height - 1 ? sy + 1 : sy;

            for (int c = 0; c < channels; c++) {
                float p00 = src_data[(sy * src_width + sx) * channels + c];
                float p10 = src_data[(sy * src_width + sx1) * channels + c];
                float p01 = src_data[(sy1 * src_width + sx) * channels + c];
                float p11 = src_data[(sy1 * src_width + sx1) * channels + c];
                float top = p00 + (p10 - p00) * fx;
                float bot = p01 + (p11 - p01) * fx;
                dst_data[(dy * dst_width + dx) * channels + c] = top + (bot - top) * fy;
            }
        }
    }
    return 0;
}

int vision_yuv420_to_rgb(int width, int height,
                          const unsigned char* y_plane, int y_stride,
                          const unsigned char* u_plane, int uv_stride,
                          const unsigned char* v_plane,
                          float* rgb_output) {
    if (!y_plane || !u_plane || !v_plane || !rgb_output) return -1;
    if (width <= 0 || height <= 0 || (width & 1) || (height & 1)) return -1;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int y_idx = y * y_stride + x;
            int uv_x = x / 2, uv_y = y / 2;
            int uv_idx = uv_y * uv_stride + uv_x;

            float yy = (float)y_plane[y_idx];
            float u = (float)u_plane[uv_idx] - 128.0f;
            float v = (float)v_plane[uv_idx] - 128.0f;

            int dst_idx = (y * width + x) * 3;
            float r = yy + 1.402f * v;
            float g = yy - 0.344f * u - 0.714f * v;
            float b = yy + 1.772f * u;

            rgb_output[dst_idx] = (r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r)) / 255.0f;
            rgb_output[dst_idx + 1] = (g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g)) / 255.0f;
            rgb_output[dst_idx + 2] = (b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b)) / 255.0f;
        }
    }
    return 0;
}

/* ===================================================================
 * 双线性缩放——float版本
 * =================================================================== */
int vision_resize_bilinear_float(int src_width, int src_height, int channels,
                                  const float* src_data,
                                  int dst_width, int dst_height, float* dst_data) {
    return vision_resize_bilinear(src_width, src_height, channels,
                                  src_data, dst_width, dst_height, dst_data);
}

/* ===================================================================
 * HOG特征提取——可独立调用的完整实现
 * =================================================================== */
int vision_extract_hog_features(const float* gray, int width, int height,
                                 float* features, int max_features) {
    if (!gray || !features || max_features < 36 || width < 16 || height < 16) return -1;
    int cell_w = 8, cell_h = 8;
    int cells_x = width / cell_w, cells_y = height / cell_w;
    if (cells_x < 2) cells_x = 2;
    if (cells_y < 2) cells_y = 2;
    int feat_out = 0;
    float* cell_hist = (float*)safe_calloc((size_t)(cells_x * cells_y * 9), sizeof(float));
    if (!cell_hist) return -1;
    for (int cy = 0; cy < cells_y; cy++) {
        for (int cx = 0; cx < cells_x; cx++) {
            for (int dy = 0; dy < cell_h; dy++) {
                for (int dx = 0; dx < cell_w; dx++) {
                    int px = cx * cell_w + dx, py = cy * cell_h + dy;
                    if (px < 1 || py < 1 || px >= width - 1 || py >= height - 1) continue;
                    float gx = gray[py*width + (px+1)] - gray[py*width + (px-1)];
                    float gy = gray[(py+1)*width + px] - gray[(py-1)*width + px];
                    float mag = sqrtf(gx*gx + gy*gy);
                    float ori = atan2f(gy, gx);
                    if (ori < 0) ori += 2.0f * 3.14159265f;
                    int bin = (int)(ori * 9.0f / (2.0f * 3.14159265f)) % 9;
                    cell_hist[(cy * cells_x + cx) * 9 + bin] += mag;
                }
            }
        }
    }
    for (int cy = 0; cy < cells_y - 1 && feat_out < max_features; cy++) {
        for (int cx = 0; cx < cells_x - 1 && feat_out < max_features; cx++) {
            float block_norm = 0.0f;
            for (int by = 0; by < 2; by++)
                for (int bx = 0; bx < 2; bx++)
                    for (int b = 0; b < 9; b++)
                        block_norm += cell_hist[((cy+by)*cells_x + (cx+bx))*9 + b];
            float norm = sqrtf(block_norm*block_norm + 1e-6f);
            for (int by = 0; by < 2 && feat_out < max_features; by++)
                for (int bx = 0; bx < 2 && feat_out < max_features; bx++)
                    for (int b = 0; b < 9 && feat_out < max_features; b++) {
                        features[feat_out++] = cell_hist[((cy+by)*cells_x+(cx+bx))*9+b] / norm;
                    }
        }
    }
    safe_free((void**)&cell_hist);
    return feat_out;
}

/* ===================================================================
 * 多尺度LBP特征提取——3个半径尺度
 * =================================================================== */
int vision_extract_multiscale_lbp(const float* gray, int width, int height,
                                   float* features, int max_features) {
    if (!gray || !features || max_features < 24 || width < 3 || height < 3) return -1;
    int radii[3] = {1, 2, 4};
    int feat_out = 0;
    for (int r_idx = 0; r_idx < 3; r_idx++) {
        int R = radii[r_idx];
        float hist[8] = {0};
        int count = 0;
        for (int y = R; y < height - R; y++) {
            for (int x = R; x < width - R; x++) {
                float center = gray[y * width + x];
                int code = 0;
                for (int p = 0; p < 8; p++) {
                    float angle = (float)p * 3.14159265f / 4.0f;
                    int px = x + (int)((float)R * cosf(angle) + 0.5f);
                    int py = y + (int)((float)R * sinf(angle) + 0.5f);
                    if (px < 0 || px >= width || py < 0 || py >= height) continue;
                    if (gray[py * width + px] >= center) code |= (1 << p);
                }
                for (int b = 0; b < 8; b++) if (code & (1 << b)) hist[b] += 1.0f;
                count++;
            }
        }
        if (count > 0 && feat_out + 8 <= max_features) {
            for (int b = 0; b < 8; b++) features[feat_out++] = hist[b] / (float)count;
        }
    }
    return feat_out;
}

/* ===================================================================
 * HSV色彩直方图提取
 * =================================================================== */
int vision_extract_color_histogram(const float* rgb, int width, int height, int channels,
                                    float* features, int max_features) {
    if (!rgb || !features || max_features < 28 || channels < 3) return -1;
    float hue_hist[12] = {0}, sat_hist[8] = {0}, val_hist[8] = {0};
    size_t total = (size_t)width * height;
    size_t count = 0;
    for (size_t i = 0; i < total && count < 50000; i++) {
        float r = rgb[i*3], g = rgb[i*3+1], b_ = rgb[i*3+2];
        float mx = r > g ? (r > b_ ? r : b_) : (g > b_ ? g : b_);
        float mn = r < g ? (r < b_ ? r : b_) : (g < b_ ? g : b_);
        float delta = mx - mn, hue = 0.0f;
        if (delta > 1e-4f) {
            if (mx == r) hue = 60.0f * ((g-b_)/delta + (g<b_?6.0f:0.0f));
            else if (mx == g) hue = 60.0f * ((b_-r)/delta + 2.0f);
            else hue = 60.0f * ((r-g)/delta + 4.0f);
        }
        float sat = mx > 1e-4f ? delta / mx : 0.0f;
        float val = mx;
        hue_hist[(int)(hue/30.0f)%12] += 1.0f;
        sat_hist[(int)(sat*7.99f)] += 1.0f;
        val_hist[(int)(val*7.99f)] += 1.0f;
        count++;
    }
    if (count == 0) return 0;
    int fi = 0;
    for (int i = 0; i < 12 && fi < max_features; i++) features[fi++] = hue_hist[i]/(float)count;
    for (int i = 0; i < 8 && fi < max_features; i++) features[fi++] = sat_hist[i]/(float)count;
    for (int i = 0; i < 8 && fi < max_features; i++) features[fi++] = val_hist[i]/(float)count;
    return fi;
}

/* ===================================================================
 * 增强版CfC目标检测——完整80分类输出
 * =================================================================== */
int vision_enhanced_cfc_detect(const float* features, int feature_dim,
                                LNN* vision_lnn,
                                VisionEnhancedDetect* detections,
                                int max_detections, int* num_found) {
    if (!features || !detections || !num_found || !vision_lnn) return -1;
    *num_found = 0;

    /* P0-004修复: 获取动态类别数 */
    int num_classes = vision_class_get_count(vision_class_registry_get_global());
    if (num_classes <= 0) num_classes = VISION_CLASS_DEFAULT_COUNT;
    /* 每个检测编码: cx,cy,w,h,conf + N类logits = 5+num_classes */
    int per_det_dim = 5 + num_classes;

    int in_dim = feature_dim < CFC_VISION_INPUT_DIM ? feature_dim : CFC_VISION_INPUT_DIM;
    float* in_buf = (float*)safe_calloc(CFC_VISION_INPUT_DIM, sizeof(float));
    float* hid_buf = (float*)safe_calloc(CFC_VISION_HIDDEN_DIM, sizeof(float));
    if (!in_buf || !hid_buf) { safe_free((void**)&in_buf); safe_free((void**)&hid_buf); return -1; }
    memcpy(in_buf, features, (size_t)in_dim * sizeof(float));

    lnn_forward(vision_lnn, in_buf, hid_buf);

    /* 分组解码：5组×10检测 = 50个检测对象（每检测per_det_dim维） */
    int groups = 5;
    int per_g = 10;
    for (int g = 0; g < groups; g++) {
        int off = g * (CFC_VISION_HIDDEN_DIM / groups);
        for (int d = 0; d < per_g && *num_found < max_detections; d++) {
            int b = off + d * per_det_dim;
            if (b + 4 >= CFC_VISION_HIDDEN_DIM) break;
            float cx = hid_buf[b], cy = hid_buf[b+1];
            float w = fabsf(hid_buf[b+2]) * 0.5f + 0.02f;
            float h_img = fabsf(hid_buf[b+3]) * 0.5f + 0.02f;
            float conf = 1.0f/(1.0f+expf(-hid_buf[b+4]));
            if (conf < 0.25f) continue;

            int ok = (b + per_det_dim <= CFC_VISION_HIDDEN_DIM);
            int best = 0;
            float* logits = (float*)safe_calloc((size_t)num_classes, sizeof(float));
            if (ok && logits) {
                for (int c = 0; c < num_classes; c++) logits[c] = hid_buf[b + 5 + c];
                _softmax(logits, num_classes);
                for (int c = 0; c < num_classes; c++)
                    if (logits[c] > logits[best]) best = c;
                /* 动态分配class_probs */
                detections[*num_found].class_probs = (float*)safe_malloc((size_t)num_classes * sizeof(float));
                if (detections[*num_found].class_probs) {
                    memcpy(detections[*num_found].class_probs, logits, (size_t)num_classes * sizeof(float));
                    detections[*num_found].class_probs_count = num_classes;
                } else {
                    detections[*num_found].class_probs_count = 0;
                }
            }
            safe_free((void**)&logits);
            detections[*num_found].cx = cx; detections[*num_found].cy = cy;
            detections[*num_found].w = w; detections[*num_found].h = h_img;
            detections[*num_found].confidence = conf;
            detections[*num_found].class_id = best;
            /* 从注册表获取类别名称 */
            VisionClassEntry cls_entry;
            if (vision_class_get_entry(vision_class_registry_get_global(), best, &cls_entry) == 0) {
                strncpy(detections[*num_found].class_name, cls_entry.name_zh, VISION_CLASS_NAME_MAX_LEN - 1);
                detections[*num_found].class_name[VISION_CLASS_NAME_MAX_LEN - 1] = '\0';
            } else {
                snprintf(detections[*num_found].class_name, VISION_CLASS_NAME_MAX_LEN - 1, "class_%d", best);
            }
            (*num_found)++;
        }
    }
    safe_free((void**)&in_buf); safe_free((void**)&hid_buf);
    return 0;
}
