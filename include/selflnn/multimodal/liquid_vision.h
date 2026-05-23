/**
 * @file liquid_vision.h
 * @brief 统一液态视觉处理系统
 *
 * 基于CfC（Closed-form Continuous-time）液态神经网络的统一视觉处理组件。
 * 所有视觉处理在单一液态神经网络框架内完成：
 * τ dh/dt = -h + σ(W_gx·x + W_gh·h + b_g) ⊙ tanh(W_ax·x + W_ah·h + b_a)
 *
 * 不引入任何Transformer、注意力机制或独立处理器。
 * 纯C实现，不依赖任何第三方库。
 *
 * 模块结构：
 *   【主路径】LiquidVisionManager（PatchEncoder → SpatialProcessor → CfCEvolver）
 *   【辅助路径】传统CV预处理（Sobel/LBP/HOG/HSV直方图等）
 *   【检测路径】YOLO风格目标检测头 + NMS
 *   【类别系统】动态可扩展视觉类别注册表（80类COCO默认 + 动态扩展）
 *   【兼容层】CfcVisionProcessor/CfcOdeLayer（保持现有模块兼容）
 *
 * 本文件整合了原 vision.h 和 deep_vision.h 的所有接口。
 */

#ifndef SELFLNN_LIQUID_VISION_H
#define SELFLNN_LIQUID_VISION_H

#include <stddef.h>
#include "selflnn/core/lnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * 第一部分：液态补丁编码器（原liquid_vision.h）
 * =================================================================== */

typedef struct {
    int patch_size;
    int stride;
    int max_patches;
    int input_channels;
    int patch_hidden_dim;
    float time_constant;
    float delta_t;
    int use_adaptive_tau;
    float min_tau;
    float max_tau;
    int enable_noise;
    float noise_std;
} LiquidPatchEncoderConfig;

typedef struct LiquidPatchEncoder LiquidPatchEncoder;

LiquidPatchEncoderConfig liquid_patch_encoder_get_default_config(void);
LiquidPatchEncoder* liquid_patch_encoder_create(const LiquidPatchEncoderConfig* config);
void liquid_patch_encoder_free(LiquidPatchEncoder* encoder);

int liquid_patch_encoder_forward(LiquidPatchEncoder* encoder,
                                 int width, int height, int channels,
                                 const float* image_data,
                                 float* patch_embeddings, int max_patches);
int liquid_patch_encoder_backward(LiquidPatchEncoder* encoder,
                                   const float* dL_dembeddings, int num_patches,
                                   float* dL_dimage, int image_size,
                                   float learning_rate);
void liquid_patch_encoder_reset(LiquidPatchEncoder* encoder);
int liquid_patch_encoder_get_num_patches(const LiquidPatchEncoder* encoder,
                                          int width, int height);
int liquid_patch_encoder_get_hidden_dim(const LiquidPatchEncoder* encoder);

/* ===================================================================
 * 第二部分：液态视觉CfC演化器（原liquid_vision.h）
 * =================================================================== */

typedef struct {
    int visual_state_dim;
    int num_visual_channels;
    float base_time_constant;
    float delta_t;
    int use_adaptive_tau;
    float min_tau;
    float max_tau;
    int enable_cross_channel;
    int enable_noise;
    float noise_std;
} LiquidVisualCfCEvolverConfig;

typedef struct LiquidVisualCfCEvolver LiquidVisualCfCEvolver;

LiquidVisualCfCEvolverConfig liquid_visual_cfc_evolver_get_default_config(void);
LiquidVisualCfCEvolver* liquid_visual_cfc_evolver_create(const LiquidVisualCfCEvolverConfig* config);
void liquid_visual_cfc_evolver_free(LiquidVisualCfCEvolver* evolver);

int liquid_visual_cfc_evolver_forward(LiquidVisualCfCEvolver* evolver,
                                       const float* visual_input, int input_dim,
                                       float* visual_state);
int liquid_visual_cfc_evolver_step(LiquidVisualCfCEvolver* evolver,
                                    const float* visual_inputs, int input_dim,
                                    int num_steps, float* final_state);
void liquid_visual_cfc_evolver_reset(LiquidVisualCfCEvolver* evolver);
int liquid_visual_cfc_evolver_get_state_dim(const LiquidVisualCfCEvolver* evolver);

/* ===================================================================
 * 第三部分：液态空间特征处理器（原liquid_vision.h）
 * =================================================================== */

typedef struct {
    int num_features;
    int feature_dim;
    int spatial_hidden_dim;
    float time_constant;
    float delta_t;
    int use_adaptive_tau;
    int num_evolution_steps;
    float min_tau;
    float max_tau;
} LiquidSpatialProcessorConfig;

typedef struct LiquidSpatialProcessor LiquidSpatialProcessor;

LiquidSpatialProcessorConfig liquid_spatial_processor_get_default_config(void);
LiquidSpatialProcessor* liquid_spatial_processor_create(const LiquidSpatialProcessorConfig* config);
void liquid_spatial_processor_free(LiquidSpatialProcessor* processor);

int liquid_spatial_processor_forward(LiquidSpatialProcessor* processor,
                                      const float* features, int num_features, int feature_dim,
                                      float* output);
void liquid_spatial_processor_reset(LiquidSpatialProcessor* processor);
int liquid_spatial_processor_get_output_dim(const LiquidSpatialProcessor* processor);

/* ===================================================================
 * 第四部分：液态视觉管理器（原liquid_vision.h）
 * =================================================================== */

typedef enum {
    LIQUID_VISION_PATCH_ENCODER = 0,
    LIQUID_VISION_CFC_EVOLVER = 1,
    LIQUID_VISION_SPATIAL = 2,
    LIQUID_VISION_FULL = 3
} LiquidVisionPipelineType;

typedef struct {
    LiquidVisionPipelineType pipeline_type;
    LiquidPatchEncoderConfig patch_config;
    LiquidVisualCfCEvolverConfig evolver_config;
    LiquidSpatialProcessorConfig spatial_config;
    int enable_temporal_integration;
    int temporal_window_size;
} LiquidVisionManagerConfig;

typedef struct LiquidVisionManager LiquidVisionManager;

LiquidVisionManagerConfig liquid_vision_manager_get_default_config(void);
LiquidVisionManager* liquid_vision_manager_create(const LiquidVisionManagerConfig* config);
void liquid_vision_manager_free(LiquidVisionManager* manager);

int liquid_vision_manager_forward(LiquidVisionManager* manager,
                                   int width, int height, int channels,
                                   const float* image_data,
                                   float* output, int output_dim);
int liquid_vision_manager_set_features(LiquidVisionManager* manager,
                                        const float* features, int feature_dim);
void liquid_vision_manager_reset(LiquidVisionManager* manager);
int liquid_vision_manager_get_output_dim(const LiquidVisionManager* manager);
int liquid_vision_manager_save(const LiquidVisionManager* manager, const char* filepath);
int liquid_vision_manager_load(LiquidVisionManager* manager, const char* filepath);

int liquid_vision_manager_enable_cfc_temporal(LiquidVisionManager* manager, int enable);
int liquid_vision_manager_reset_temporal(LiquidVisionManager* manager);
LNN* liquid_vision_manager_get_lnn(LiquidVisionManager* manager);
int liquid_vision_manager_set_lnn(LiquidVisionManager* manager, LNN* lnn);

/* ===================================================================
 * 第五部分：动态可扩展视觉类别注册表（原vision.h）
 * =================================================================== */

#define VISION_CLASS_DEFAULT_COUNT 80
#define VISION_CLASS_MAX_CAPACITY  100000
#define VISION_CLASS_NAME_MAX_LEN   64

typedef struct VisionClassRegistry VisionClassRegistry;

typedef struct {
    int class_id;
    char name_zh[VISION_CLASS_NAME_MAX_LEN];
    char name_en[VISION_CLASS_NAME_MAX_LEN];
    int is_learned;
    int sample_count;
    float confidence_threshold;
} VisionClassEntry;

VisionClassRegistry* vision_class_registry_create(void);
void vision_class_registry_free(VisionClassRegistry* registry);
VisionClassRegistry* vision_class_registry_get_global(void);
int vision_class_register(VisionClassRegistry* registry,
                          const char* name_zh, const char* name_en);
int vision_class_get_count(const VisionClassRegistry* registry);
int vision_class_get_entry(const VisionClassRegistry* registry, int class_id,
                           VisionClassEntry* entry);
int vision_class_add_samples(VisionClassRegistry* registry, int class_id, int count);
const char* vision_get_class_name_zh(int class_id);
const char* vision_get_class_name_en(int class_id);

/* ===================================================================
 * 第六部分：检测结构体 + NMS（原vision.h）
 * =================================================================== */

typedef struct {
    float cx, cy, w, h;
    float confidence;
    int class_id;
    char class_name[VISION_CLASS_NAME_MAX_LEN];
    float* class_probs;
    int class_probs_count;
} CfCVisionDetection;

int vision_nms(CfCVisionDetection* detections, int count, float iou_threshold);
void vision_free_detections(CfCVisionDetection* detections, int count);

/* ===================================================================
 * 第七部分：图像处理工具函数（原vision.h）
 * =================================================================== */

int vision_resize_bilinear(int src_width, int src_height, int channels,
                            const float* src_data,
                            int dst_width, int dst_height, float* dst_data);
int vision_resize_bilinear_float(int src_width, int src_height, int channels,
                            const float* src_data,
                            int dst_width, int dst_height, float* dst_data);
int vision_yuv420_to_rgb(int width, int height,
                          const unsigned char* y_plane, int y_stride,
                          const unsigned char* u_plane, int uv_stride,
                          const unsigned char* v_plane,
                          float* rgb_output);

/* ===================================================================
 * 第八部分：传统CV特征提取器（原vision.h）- 辅助预处理路径
 * =================================================================== */

int vision_extract_hog_features(const float* gray, int width, int height,
                                float* features, int max_features);
int vision_extract_multiscale_lbp(const float* gray, int width, int height,
                                  float* features, int max_features);
int vision_extract_color_histogram(const float* rgb, int width, int height, int channels,
                                   float* features, int max_features);

/* ===================================================================
 * 第九部分：CfC ODE层（原deep_vision.h）- 兼容层
 * =================================================================== */

typedef struct {
    int input_dim;
    int hidden_dim;
    int num_layers;
    float time_constant;
    float delta_t;
    int use_adaptive_tau;
    float min_tau;
    float max_tau;
    int use_bias;
    int use_liquid_gate;
    float noise_std;
} CfcOdeLayerConfig;

typedef struct CfcOdeLayer CfcOdeLayer;

CfcOdeLayerConfig cfc_ode_layer_get_default_config(void);
CfcOdeLayer* cfc_ode_layer_create(const CfcOdeLayerConfig* config);
void cfc_ode_layer_free(CfcOdeLayer* layer);
int cfc_ode_layer_forward(CfcOdeLayer* layer, const float* input, float* output);
int cfc_ode_layers_forward(CfcOdeLayer** layers, int num_layers,
                           const float* input, float* output);
int cfc_ode_layer_backward(CfcOdeLayer* layer,
                           const float* dL_doutput,
                           const float* input,
                           const float* output,
                           float* dL_dinput,
                           float learning_rate);

/* ===================================================================
 * 第十部分：CfcVisionProcessor兼容层（原deep_vision.h）
 * =================================================================== */

typedef struct {
    int image_width;
    int image_height;
    int image_channels;
    int patch_size;
    int output_dim;
    int num_ode_layers;
    float time_constant;
    float delta_t;
} CfcVisionConfig;

typedef struct CfcVisionProcessor CfcVisionProcessor;

CfcVisionConfig cfc_vision_get_default_config(void);
CfcVisionProcessor* cfc_vision_processor_create(const CfcVisionConfig* config);
void cfc_vision_processor_destroy(CfcVisionProcessor* processor);

int cfc_vision_extract_features(CfcVisionProcessor* processor,
                                const float* image_data,
                                int width, int height, int channels,
                                float* features, size_t max_features);
void cfc_vision_get_statistics(CfcVisionProcessor* processor,
                                size_t* total_operations,
                                float* memory_usage_mb,
                                float* average_time_ms);
int cfc_vision_save_processor(CfcVisionProcessor* processor, const char* filename);
CfcVisionProcessor* cfc_vision_load_processor(const char* filename);

/* ===================================================================
 * 第十一部分：目标检测API（原deep_vision.h）
 * =================================================================== */

#define CFC_VISION_MAX_CLASSES 1000
#define CFC_VISION_MAX_DETECTIONS 100

typedef struct {
    int class_id;
    float confidence;
    float x, y;
    float width, height;
} CfcDetectionResult;

typedef struct {
    int num_classes;
    float conf_threshold;
    float iou_threshold;
    int max_detections;
} CfcDetectionConfig;

CfcDetectionConfig cfc_vision_get_default_detection_config(void);

int cfc_vision_detect(CfcVisionProcessor* processor,
                       const float* image_data,
                       int width, int height, int channels,
                       CfcDetectionResult* results, int max_results,
                       int* num_detected);
int cfc_vision_set_detection_threshold(CfcVisionProcessor* processor,
                                        float conf_threshold,
                                        float iou_threshold);
int cfc_vision_get_detection_stats(CfcVisionProcessor* processor,
                                    int* total_detections,
                                    float* avg_confidence);
void cfc_vision_mark_trained(CfcVisionProcessor* processor);
int cfc_vision_is_trained(const CfcVisionProcessor* processor);
int cfc_vision_train_network(CfcVisionProcessor* processor,
    const float* training_images, const float* target_features,
    int num_samples, int num_epochs, float learning_rate);

/* ===================================================================
 * 第十二部分：统一视觉入口（整合后的主接口）
 * =================================================================== */

typedef struct {
    int target_width;
    int target_height;
    int grayscale;
    int feature_dimension;
    int enable_cfc;
    size_t cfc_hidden_size;
    float cfc_time_constant;
    int enable_multiscale_pyramid;
    int enable_hog;
    int enable_color_histogram;
} LiquidVisionConfig;

typedef struct LiquidVisionProcessor LiquidVisionProcessor;

LiquidVisionProcessor* liquid_vision_processor_create(const LiquidVisionConfig* config);
void liquid_vision_processor_free(LiquidVisionProcessor* processor);

int liquid_vision_process_image(LiquidVisionProcessor* processor,
                        int width, int height, int channels,
                        const float* data,
                        float* features, size_t max_features);
int liquid_vision_processor_get_config(const LiquidVisionProcessor* processor,
                                        LiquidVisionConfig* config);
int liquid_vision_processor_set_config(LiquidVisionProcessor* processor,
                                        const LiquidVisionConfig* config);
void liquid_vision_processor_reset(LiquidVisionProcessor* processor);
void liquid_vision_processor_set_lnn(LiquidVisionProcessor* processor, LNN* lnn);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LIQUID_VISION_H */
