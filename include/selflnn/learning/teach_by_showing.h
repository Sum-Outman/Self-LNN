#ifndef SELFLNN_TEACH_BY_SHOWING_H
#define SELFLNN_TEACH_BY_SHOWING_H

#include "selflnn/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TEACH_MAX_DEMOS 256
#define TEACH_MAX_STEPS_PER_DEMO 8192
#define TEACH_MAX_TASKS 64
#define TEACH_LABEL_LEN 128
#define TEACH_MODALITY_DIM 2048

typedef enum {
    TEACH_TASK_CUSTOM = 0,
    TEACH_TASK_GRASP = 1,
    TEACH_TASK_PICK_PLACE = 2,
    TEACH_TASK_ASSEMBLE = 3,
    TEACH_TASK_WALK = 4,
    TEACH_TASK_NAVIGATE = 5,
    TEACH_TASK_DRAW = 6,
    TEACH_TASK_WRITE = 7,
    TEACH_TASK_PLAY = 8,
    TEACH_TASK_COOK = 9,
    TEACH_TASK_CLEAN = 10,
    TEACH_TASK_MAX = 64
} TeachTaskType;

typedef struct TeachDemoSet {
    size_t num_demos;
    size_t max_steps;
    size_t obs_dim;
    size_t act_dim;
    float* observations;
    float* actions;
    float* rewards;
    float* task_embeddings;
    char** labels;
    TeachTaskType* task_types;
    float* timestamps;
    float* confidence_scores;
    size_t* trajectory_lengths;
} TeachDemoSet;

typedef struct {
    float learning_rate;
    float behavior_clone_weight;
    float irl_weight;
    float entropy_weight;
    size_t batch_size;
    size_t num_epochs;
    int use_attention;
    int use_bc_loss;
    int use_irl_loss;
    float max_grad_norm;
    float demo_noise_std;
    int augment_demos;
    int use_mirror_augmentation;
} TeachTrainConfig;

#define TEACH_TRAIN_CONFIG_DEFAULT { \
    0.001f, 1.0f, 0.1f, 0.01f, 64, 100, 1, 1, 0, 5.0f, 0.01f, 1, 1 \
}

typedef enum {
    TEACH_ENCODE_NONE = 0,
    TEACH_ENCODE_CNN = 1,
    TEACH_ENCODE_LNN = 2,
    TEACH_ENCODE_HYBRID = 3
} TeachEncodeType;

typedef struct {
    float trajectory_match_score;
    float task_recognition_confidence;
    float reproduction_accuracy;
    float generalization_score;
    size_t num_tasks_learned;
    size_t total_demonstrations;
    float avg_learning_time;
} TeachSystemStats;

typedef struct TeachSystem TeachSystem;

TeachSystem* teach_system_create(size_t obs_dim, size_t act_dim,
                                 TeachTrainConfig config);

void teach_system_destroy(TeachSystem* system);

int teach_record_demonstration(TeachSystem* system,
                                const float* observations,
                                const float* actions,
                                size_t num_steps,
                                size_t obs_dim,
                                size_t act_dim,
                                const char* label,
                                TeachTaskType task_type);

int teach_encode_demonstrations(TeachSystem* system,
                                 TeachEncodeType encode_type);

int teach_train_from_demos(TeachSystem* system,
                            int (*progress_callback)(float progress,
                                                     float loss,
                                                     void* user_data),
                            void* user_data);

int teach_reproduce_task(TeachSystem* system,
                          const char* task_label,
                          float* current_obs,
                          size_t obs_dim,
                          float* actions_out,
                          size_t* num_steps_out,
                          float temperature);

int teach_recognize_task(TeachSystem* system,
                          const float* observations,
                          size_t num_steps,
                          size_t obs_dim,
                          char* task_label_out,
                          size_t label_buf_size,
                          float* confidence_out);

void teach_get_stats(TeachSystem* system, TeachSystemStats* stats);

int teach_export_demos(TeachSystem* system,
                        const char* file_path);

int teach_import_demos(TeachSystem* system,
                        const char* file_path);

int teach_clear_demos(TeachSystem* system,
                       size_t keep_latest);

int teach_evaluate_reproduction(TeachSystem* system,
                                 const char* task_label,
                                 const float* ground_truth_actions,
                                 size_t num_steps,
                                 size_t act_dim,
                                 float* accuracy_out,
                                 float* similarity_out);

int teach_incremental_update(TeachSystem* system,
                              const float* new_observations,
                              const float* new_actions,
                              size_t num_steps,
                              size_t obs_dim,
                              size_t act_dim,
                              const char* label,
                              float learning_rate_scale);

TeachDemoSet* teach_get_demo_set(TeachSystem* system);
void teach_free_demo_set(TeachDemoSet* set);

/* ==================== 实物教学5大核心算法 ==================== */

#define TEACH_VISUAL_DIM   512
#define TEACH_AUDIO_DIM    256
#define TEACH_TACTILE_DIM  128
#define TEACH_TEXT_DIM     256
#define TEACH_UNIFIED_DIM  1024
#define TEACH_MAX_CONCEPTS 1024
#define TEACH_MAX_PROPERTIES 64

typedef enum {
    TEACH_CATEGORY_OBJECT = 0,
    TEACH_CATEGORY_ACTION = 1,
    TEACH_CATEGORY_PROPERTY = 2,
    TEACH_CATEGORY_QUANTITY = 3,
    TEACH_CATEGORY_RELATION = 4,
    TEACH_CATEGORY_MAX = 16
} TeachConceptCategory;

typedef struct {
    char concept_name[TEACH_LABEL_LEN];
    float visual_embedding[TEACH_VISUAL_DIM];
    float audio_embedding[TEACH_AUDIO_DIM];
    float tactile_embedding[TEACH_TACTILE_DIM];
    float text_embedding[TEACH_TEXT_DIM];
    float unified_embedding[TEACH_UNIFIED_DIM];
    float physical_properties[TEACH_MAX_PROPERTIES];
    int num_properties;
    int num_examples;
    float confidence;
    TeachConceptCategory category;
    int is_abstract;
    float abstraction_level;
} TeachConcept;

/**
 * @brief 多感官绑定: 将视觉+语音+触觉+文本同时绑定到统一概念
 *
 * 核心算法: 多模态特征提取→跨模态对齐→统一表征空间映射→概念创建/更新
 */
int teach_bind_concept(TeachSystem* system,
                        const float* visual_feat, size_t visual_dim,
                        const float* audio_feat, size_t audio_dim,
                        const float* tactile_feat, size_t tactile_dim,
                        const char* concept_name);

/**
 * @brief Look-and-Learn(看一眼就认识): 单样本视觉概念学习
 *
 * 核心算法: 视觉捕获→特征提取→原型网络匹配→概念确认/创建
 * 支持一眼认出已知概念或新建未知概念
 */
int teach_look_and_learn(TeachSystem* system,
                          const float* visual_data, size_t width, size_t height,
                          const char* concept_name,
                          float* concept_embedding, size_t embed_dim);

/**
 * @brief Say-and-Associate(说出来就关联): 语音/文本关联学习
 *
 * 核心算法: 语音识别→文本解析→语义嵌入→跨模态关联→概念强化
 * 教师说出概念名称，系统自动关联到当前视觉/触觉上下文
 */
int teach_say_and_associate(TeachSystem* system,
                             const float* audio_data, size_t audio_len,
                             const char* text,
                             const float* context_visual, size_t visual_dim,
                             const char* concept_name,
                             float* association_embedding, size_t embed_dim);

/**
 * @brief Touch-and-Understand(摸一下就知道属性): 物理属性感知学习
 *
 * 核心算法: 触觉/力传感器→物理特征提取(硬度/温度/纹理/重量)→
 * 属性向量生成→概念属性绑定→多示例统计泛化
 */
int teach_touch_and_understand(TeachSystem* system,
                                const float* sensor_data, size_t sensor_dim,
                                const char* concept_name,
                                float* property_vector, size_t prop_dim);

/**
 * @brief Count-and-Generalize(数数后概括数量概念): 数量概念抽象
 *
 * 核心算法: 视觉序列分析→个体识别→计数→数量→数量概念抽象→
 * 跨情境泛化(不同物体相同数量也能识别)
 */
int teach_count_and_generalize(TeachSystem* system,
                                const float* visual_sequence, size_t num_frames,
                                size_t width, size_t height,
                                const char* count_concept, int count_value,
                                float* abstraction_embedding, size_t embed_dim);

/**
 * @brief 测试验证: 展示物品/声音/触感，系统识别概念
 *
 * 通过多模态输入验证已学概念，返回识别结果和置信度
 */
int teach_test_concept(TeachSystem* system,
                        const float* visual_feat, size_t visual_dim,
                        const float* audio_feat, size_t audio_dim,
                        const float* tactile_feat, size_t tactile_dim,
                        char* recognized_concept, size_t buf_size,
                        float* confidence);

/**
 * @brief 获取已学习的概念列表
 */
int teach_get_concepts(TeachSystem* system,
                        TeachConcept* concepts_out, size_t* num_concepts);

/**
 * @brief 清除指定概念
 */
int teach_clear_concept(TeachSystem* system, const char* concept_name);

/**
 * @brief 清除所有概念
 */
int teach_clear_all_concepts(TeachSystem* system);

#ifdef __cplusplus
}
#endif

#endif
