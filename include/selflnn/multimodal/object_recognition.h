/**
 * @file object_recognition.h
 * @brief 增强物体识别与场景理解接口
 */

#ifndef SELFLNN_OBJECT_RECOGNITION_H
#define SELFLNN_OBJECT_RECOGNITION_H

#include <stddef.h>

/* 前向声明，避免循环依赖 */
typedef struct DepthEstimator DepthEstimator;

#ifdef __cplusplus
extern "C" {
#endif

#define OR_MAX_OBJECTS 256
#define OR_MAX_CATEGORIES 128
#define OR_MAX_ATTRIBUTES 32

typedef struct {
    float x, y, width, height;
    float confidence;
    int category_id;
    char category_name[64];
    float features[128];
    int feature_dim;
} DetectedObject;

typedef struct {
    float hue_mean, hue_std;
    float saturation_mean;
    float brightness_mean;
    float texture_roughness;
    float texture_direction;
    char material[32];
    char shape[32];
    char size_category[16];
} ObjectAttributes;

typedef enum {
    SCENE_INDOOR = 0,
    SCENE_OUTDOOR = 1,
    SCENE_INDUSTRIAL = 2,
    SCENE_URBAN = 3,
    SCENE_NATURE = 4,
    SCENE_UNKNOWN = 5
} SceneType;

typedef struct {
    DetectedObject objects[OR_MAX_OBJECTS];
    int object_count;
    SceneType scene_type;
    char scene_name[64];
    float scene_confidence;
    int** object_relations;
    int* relation_counts;
    float dominant_color[3];
    float brightness;
    float complexity;
    int has_motion;
} SceneContext;

typedef struct ObjectRecognizer ObjectRecognizer;

ObjectRecognizer* object_recognizer_create(void);
void object_recognizer_free(ObjectRecognizer* or_obj);

/* 物体检测 */
int or_detect_objects(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch, DetectedObject* out, int max_count);
int or_classify_object(ObjectRecognizer* or_obj, const float* features, int dim, int* category_id, float* confidence);

/* 属性识别 */
int or_recognize_attributes(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch, const DetectedObject* obj, ObjectAttributes* attrs);
int or_detect_color(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch, const DetectedObject* obj, float* color_rgb);
/* RGB值→中文颜色名称映射（12种基准颜色+置信度） */
int or_get_color_name(const float* color_rgb, char* name_buf, size_t buf_size, float* confidence);
int or_estimate_size(ObjectRecognizer* or_obj, const DetectedObject* obj, float* width_m, float* height_m);

/* 场景理解 */
int or_classify_scene(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch, SceneType* type, char* name, size_t name_len);
int or_detect_relations(ObjectRecognizer* or_obj, const DetectedObject* objects, int count, SceneContext* ctx);
int or_detect_changes(ObjectRecognizer* or_obj, const float* prev, const float* curr, int w, int h, int ch, float* change_map);

/* 分类器训练 */
int or_train_classifier(ObjectRecognizer* or_obj, const float* features, const int* labels, int samples, int dim, int categories);
int or_save_model(const ObjectRecognizer* or_obj, const char* filepath);
int or_load_model(ObjectRecognizer* or_obj, const char* filepath);

/* CfC深度学习训练接口 —— 独立CfC管道训练，不影响HOG模板 */
int or_train_cfc(ObjectRecognizer* or_obj, const float* features, const int* labels, int samples, int dim, int categories, int epochs, float lr);
int object_cfc_train(ObjectRecognizer* or_obj, const float* features, const int* labels, int samples, int dim, int categories, int epochs, float lr);

/* 深度信息集成 —— 解决 H-010 硬编码物距问题 */
int or_set_depth_estimator(ObjectRecognizer* or_obj, DepthEstimator* estimator);
int or_set_depth_map(ObjectRecognizer* or_obj, const float* depth_map, int w, int h);
int or_clear_depth_map(ObjectRecognizer* or_obj);

#ifdef __cplusplus
}
#endif
#endif
