#ifndef SELFLNN_IMAGE_RECOGNITION_DEEP_H
#define SELFLNN_IMAGE_RECOGNITION_DEEP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IRD_MAX_PARTS 32
#define IRD_MAX_FINE_CATEGORIES 256
#define IRD_ATTRIBUTE_DIM 128
#define IRD_SEMANTIC_DIM 256
#define IRD_MAX_SUPPORT_SAMPLES 256
#define IRD_MAX_PROTOTYPES 256
#define IRD_MAX_PATCHES 256

/* ======================================================================== */
/*  细粒度分类                                                              */
/* ======================================================================== */

typedef enum {
    IRD_PART_NONE = 0, IRD_PART_HEAD = 1, IRD_PART_BODY = 2,
    IRD_PART_WING = 3, IRD_PART_TAIL = 4, IRD_PART_LEG = 5,
    IRD_PART_WHEEL = 6, IRD_PART_DOOR = 7, IRD_PART_WINDOW = 8,
    IRD_PART_ENGINE = 9, IRD_PART_OTHER = 10
} IRDPartType;

typedef struct {
    int part_id;
    IRDPartType part_type;
    float x, y, width, height;
    float feature_vector[IRD_SEMANTIC_DIM];
    float discriminative_score;
    float saliency_map[IRD_MAX_PATCHES];
    int num_salient_patches;
} IRDDiscriminativePart;

typedef struct {
    int fine_category_id;
    char fine_category_name[64];
    int coarse_category_id;
    char coarse_category_name[64];
    float fine_confidence;
    float coarse_confidence;
    IRDDiscriminativePart parts[IRD_MAX_PARTS];
    int num_parts;
    float global_feature[IRD_SEMANTIC_DIM];
    float local_feature[IRD_SEMANTIC_DIM];
    float bilinear_feature[IRD_SEMANTIC_DIM];
} IRDFineGrainedResult;

typedef struct {
    int input_dim;
    int num_fine_categories;
    int num_coarse_categories;
    int num_parts;
    int patch_size;
    int feature_dim;
    float cfc_time_constant;
    float cfc_delta_t;
    int enable_bilinear_pooling;
    float bilinear_normalization;
    int enable_part_refinement;
    int max_refinement_iterations;
    float discriminative_threshold;
    int hierarchical_taxonomy[IRD_MAX_FINE_CATEGORIES];
    char coarse_names[64][64];
    char fine_names[IRD_MAX_FINE_CATEGORIES][64];
} IRDFineConfig;

typedef struct IRDFineClassifier IRDFineClassifier;

IRDFineConfig ird_fine_get_default_config(void);
IRDFineClassifier* ird_fine_create(const IRDFineConfig* config);
void ird_fine_free(IRDFineClassifier* classifier);
int ird_fine_classify(IRDFineClassifier* classifier, const float* image, int w, int h, int ch, IRDFineGrainedResult* result);
int ird_fine_classify_batch(IRDFineClassifier* classifier, const float* images, int w, int h, int ch, int batch_size, IRDFineGrainedResult* results);
int ird_fine_locate_parts(IRDFineClassifier* classifier, const float* image, int w, int h, int ch, IRDDiscriminativePart* parts, int* num_parts);
int ird_fine_train(IRDFineClassifier* classifier, const float* images, const int* labels, int n, int w, int h, int ch, int epochs, float lr);
int ird_fine_save(const IRDFineClassifier* classifier, const char* path);
int ird_fine_load(IRDFineClassifier* classifier, const char* path);

/**
 * @brief 标记细粒度分类器为已训练
 * 在系统加载检查点或完成引导训练后调用
 * @param classifier 分类器句柄
 */
void ird_fine_mark_trained(IRDFineClassifier* classifier);
int ird_fine_is_trained(const IRDFineClassifier* classifier);

/* ======================================================================== */
/*  开放集识别                                                              */
/* ======================================================================== */

typedef struct {
    int top_predictions[256];
    float top_confidence[256];
    int is_unknown;
    float open_set_score;
    int assigned_class;
    float confidence;
    float distance_to_known;
    float nndr_score;
    float evt_score;
    int top_k;
} IRDOpenSetPrediction;

typedef struct {
    int feature_dim;
    int input_dim;
    int num_known;
    float cfc_time_constant;
    float cfc_delta_t;
    float rejection_threshold;
    float nndr_threshold;
    float rbf_gamma;
    int weibull_fit_size;
    float temperature;
} IRDOpenSetConfig;

typedef struct IRDOpenSetRecognizer IRDOpenSetRecognizer;

IRDOpenSetConfig ird_open_set_get_default_config(void);
IRDOpenSetRecognizer* ird_open_set_create(const IRDOpenSetConfig* config);
void ird_open_set_free(IRDOpenSetRecognizer* recognizer);
int ird_open_set_predict(IRDOpenSetRecognizer* recognizer, const float* features, IRDOpenSetPrediction* prediction, int top_k);
int ird_open_set_predict_image(IRDOpenSetRecognizer* recognizer, const float* image, int w, int h, int ch, IRDOpenSetPrediction* prediction);
int ird_open_set_fit_weibull(IRDOpenSetRecognizer* recognizer, const float* known_distances, int num_samples);
int ird_open_set_learn_known(IRDOpenSetRecognizer* recognizer, const float* features, const int* labels, int n);
int ird_open_set_update_rejection(IRDOpenSetRecognizer* recognizer, float threshold);
int ird_open_set_set_threshold(IRDOpenSetRecognizer* recognizer, float threshold);
int ird_open_set_save(const IRDOpenSetRecognizer* recognizer, const char* path);
int ird_open_set_load(IRDOpenSetRecognizer* recognizer, const char* path);

/* ======================================================================== */
/*  零样本识别                                                              */
/* ======================================================================== */

typedef struct {
    int class_id;
    char class_name[64];
    float confidence;
    float semantic_embedding[IRD_SEMANTIC_DIM];
    float attribute_prediction[IRD_ATTRIBUTE_DIM];
    int top_k;
    struct { int class_id; float similarity; char class_name[64]; } top_predictions[10];
} IRDZeroShotPrediction;

typedef struct {
    int feature_dim;
    int input_dim;
    int semantic_dim;
    int attribute_dim;
    float cfc_time_constant;
    float cfc_delta_t;
    float margin;
    float learning_rate;
    int max_seen_classes;
    int max_unseen_classes;
} IRDZeroShotConfig;

typedef struct IRDZeroShotRecognizer IRDZeroShotRecognizer;

IRDZeroShotConfig ird_zero_shot_get_default_config(void);
IRDZeroShotRecognizer* ird_zero_shot_create(const IRDZeroShotConfig* config);
void ird_zero_shot_free(IRDZeroShotRecognizer* recognizer);
int ird_zero_shot_predict(IRDZeroShotRecognizer* recognizer, const float* image, int w, int h, int ch, IRDZeroShotPrediction* prediction);
int ird_zero_shot_predict_from_features(IRDZeroShotRecognizer* recognizer, const float* features, IRDZeroShotPrediction* prediction);
int ird_zero_shot_set_class_attributes(IRDZeroShotRecognizer* recognizer, const float* attributes, int num_classes, int class_offset);
int ird_zero_shot_set_semantic_prototypes(IRDZeroShotRecognizer* recognizer, const float* prototypes, int num_classes, int class_offset);
int ird_zero_shot_learn_mapping(IRDZeroShotRecognizer* recognizer, const float* visual_features, const float* semantic_features, int num_samples);
int ird_zero_shot_add_seen_class(IRDZeroShotRecognizer* recognizer, int class_id, const float* semantic_prototype);
int ird_zero_shot_add_unseen_class(IRDZeroShotRecognizer* recognizer, int class_id, const float* semantic_prototype);
int ird_zero_shot_save(const IRDZeroShotRecognizer* recognizer, const char* path);
int ird_zero_shot_load(IRDZeroShotRecognizer* recognizer, const char* path);

/* ======================================================================== */
/*  少样本识别                                                              */
/* ======================================================================== */

typedef struct {
    int class_id;
    char class_name[64];
    float confidence;
    float prototype_distance;
    float cosine_similarity;
    int top_k;
    struct { int class_id; float confidence; char class_name[64]; } top_predictions[10];
} IRDFewShotPrediction;

typedef struct {
    int embedding_dim;
    int input_dim;
    int max_support;
    int max_way;
    float cfc_time_constant;
    float cfc_delta_t;
    float finetune_lr;
    int finetune_epochs;
    int distance_metric;
} IRDFewShotConfig;

typedef struct {
    float support_features[IRD_MAX_SUPPORT_SAMPLES][IRD_SEMANTIC_DIM];
    int support_labels[IRD_MAX_SUPPORT_SAMPLES];
    float prototypes[IRD_MAX_PROTOTYPES][IRD_SEMANTIC_DIM];
    int prototype_counts[IRD_MAX_PROTOTYPES];
    int num_support;
    int num_classes;
    int feature_dim;
} IRDFewShotEpisode;

typedef struct IRDFewShotRecognizer IRDFewShotRecognizer;

IRDFewShotConfig ird_few_shot_get_default_config(void);
IRDFewShotRecognizer* ird_few_shot_create(const IRDFewShotConfig* config);
void ird_few_shot_free(IRDFewShotRecognizer* recognizer);
int ird_few_shot_set_support(IRDFewShotRecognizer* recognizer, const float* images, const int* labels, int n, int w, int h, int ch);
int ird_few_shot_set_support_features(IRDFewShotRecognizer* recognizer, const float* features, const int* labels, int n);
int ird_few_shot_compute_prototypes(IRDFewShotRecognizer* recognizer);
int ird_few_shot_predict(IRDFewShotRecognizer* recognizer, const float* image, int w, int h, int ch, IRDFewShotPrediction* prediction);
int ird_few_shot_predict_from_features(IRDFewShotRecognizer* recognizer, const float* features, IRDFewShotPrediction* prediction);
int ird_few_shot_finetune(IRDFewShotRecognizer* recognizer, const float* images, const int* labels, int n, int w, int h, int ch);
int ird_few_shot_episodic_train(IRDFewShotRecognizer* recognizer, const IRDFewShotEpisode* episodes, int num_episodes, int w, int h, int ch);
int ird_few_shot_add_to_support(IRDFewShotRecognizer* recognizer, const float* image, int label, int w, int h, int ch);
int ird_few_shot_clear_support(IRDFewShotRecognizer* recognizer);
int ird_few_shot_update_prototype(IRDFewShotRecognizer* recognizer, int label, const float* new_prototype);
int ird_few_shot_save(const IRDFewShotRecognizer* recognizer, const char* path);
int ird_few_shot_load(IRDFewShotRecognizer* recognizer, const char* path);

/* ======================================================================== */
/*  深度识别管理器                                                          */
/* ======================================================================== */

typedef enum {
    IRD_MODE_FINE_GRAINED = 0,
    IRD_MODE_OPEN_SET = 1,
    IRD_MODE_ZERO_SHOT = 2,
    IRD_MODE_FEW_SHOT = 3
} IRDRecognitionMode;

typedef struct {
    int default_image_width;
    int default_image_height;
    int default_channels;
    int max_batch_size;
    int fine_grain_enabled;
    int open_set_enabled;
    int zero_shot_enabled;
    int few_shot_enabled;
    int default_recognition_mode;
} IRDDeepManagerConfig;

typedef struct {
    int primary_mode;
    int best_class_id;
    float primary_confidence;
    int is_unknown;
    int has_fine_result;
    int has_open_set_result;
    int has_zero_shot_result;
    int has_few_shot_result;
    IRDFineGrainedResult fine_result;
    IRDOpenSetPrediction open_set_result;
    IRDZeroShotPrediction zero_shot_result;
    IRDFewShotPrediction few_shot_result;
} IRDDeepRecognitionResult;

typedef struct IRDDeepManager IRDDeepManager;

IRDDeepManagerConfig ird_deep_manager_get_default_config(void);
IRDDeepManager* ird_deep_manager_create(const IRDDeepManagerConfig* config);
void ird_deep_manager_free(IRDDeepManager* manager);
int ird_deep_manager_recognize(IRDDeepManager* manager, const float* image, int w, int h, int ch, IRDDeepRecognitionResult* result);
int ird_deep_manager_set_mode(IRDDeepManager* manager, int mode);

/* 预训练权重持久化：统一二进制格式保存/加载深度管理器所有子模型权重 */
int ird_save_model_weights(const IRDDeepManager* manager, const char* path);
int ird_load_pretrained_weights(IRDDeepManager* manager, const char* path);

#ifdef __cplusplus
}
#endif
#endif
