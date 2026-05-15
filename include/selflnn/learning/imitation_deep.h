/**
 * @file imitation_deep.h
 * @brief 模仿学习深度增强系统接口
 * 
 * 视觉演示学习完整实现：动作序列分割、关键帧提取、轨迹编码、
 * 动作泛化、逆强化学习、实时模仿。
 */

#ifndef SELFLNN_IMITATION_DEEP_H
#define SELFLNN_IMITATION_DEEP_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IM_MAX_KEYFRAMES 256
#define IM_MAX_JOINTS 32
#define IM_MAX_TRAJECTORY 4096
#define IM_MAX_ACTIONS 128

typedef struct {
    int frame_id;
    float joint_positions[IM_MAX_JOINTS];
    float joint_velocities[IM_MAX_JOINTS];
    float end_effector[3];
    float importance;
    time_t timestamp;
} ImKeyframe;

typedef struct {
    int action_id;
    char name[64];
    ImKeyframe start_frame;
    ImKeyframe end_frame;
    float start_config[IM_MAX_JOINTS];
    float target_config[IM_MAX_JOINTS];
    float duration_ms;
    int parent_action;
    float confidence;
} ImActionSegment;

typedef struct {
    int trajectory_id;
    float* waypoints;
    int waypoint_count;
    int waypoint_dim;
    float* encoded_trajectory;
    int encoded_dim;
    float smoothness;
    float efficiency;
    time_t recorded_at;
} ImTrajectory;

typedef struct {
    float* features;
    int feature_dim;
    float* reward_weights;
    int reward_dim;
    float learning_rate;
    float discount_factor;
    int max_iterations;
    int use_baseline;
} ImInverseRL;

typedef struct {
    ImKeyframe keyframes[IM_MAX_KEYFRAMES];
    int keyframe_count;
    ImActionSegment actions[IM_MAX_ACTIONS];
    int action_count;
    ImTrajectory trajectory;
    ImInverseRL irl_config;
    float generalization_radius;
    int use_adaptive_grasp;
    int initialized;
} ImDemonstration;

typedef struct ImitationDeepLearner ImitationDeepLearner;

ImitationDeepLearner* imitation_deep_create(void);
void imitation_deep_free(ImitationDeepLearner* idl);

/* 演示数据处理 */
int im_load_demonstration(ImitationDeepLearner* idl, const float* joint_data, int frames, int joints, ImDemonstration* demo);
int im_extract_keyframes(ImitationDeepLearner* idl, const ImDemonstration* demo, ImKeyframe* out, int max_count);
int im_segment_actions(ImitationDeepLearner* idl, const ImDemonstration* demo);
int im_encode_trajectory(ImitationDeepLearner* idl, const ImDemonstration* demo);
int im_generalize_action(ImitationDeepLearner* idl, const ImActionSegment* action, float* new_start, float* new_goal, ImActionSegment* generalized);

/* 逆强化学习 */
int im_irl_infer_reward(ImitationDeepLearner* idl, const ImDemonstration* demo, float* reward_weights, int dim);
int im_irl_train(ImitationDeepLearner* idl, ImDemonstration* demos, int demo_count, int iterations);

/* 实时模仿 */
int im_start_online_mimic(ImitationDeepLearner* idl);
int im_process_observation(ImitationDeepLearner* idl, const float* joint_positions, int joints, float* action_output, int action_dim);
int im_stop_online_mimic(ImitationDeepLearner* idl);
int im_get_imitation_progress(const ImitationDeepLearner* idl, float* progress);

/* 行为克隆 */
int im_behavioral_clone_train(ImitationDeepLearner* idl, const float* states, int state_dim, const float* actions, int action_dim, int samples, int epochs);
int im_behavioral_clone_predict(ImitationDeepLearner* idl, const float* state, float* action, int action_dim);

#ifdef __cplusplus
}
#endif
#endif
